//! The Jieba Chinese Word Segmentation Implemented in Rust
//!
//! ## Installation
//!
//! Add it to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! jieba-rs-yada = "0.1"
//! ```
//!
//! then you are good to go. If you are using Rust 2015 you have to ``extern crate jieba_rs_yada`` to your crate root as well.
//!
//! ## Example
//!
//! ```rust
//! use jieba_rs_yada::Jieba;
//!
//! let jieba = Jieba::new();
//! let words = jieba.cut("我们中出了一个叛徒", false);
//! assert_eq!(words, vec!["我们", "中", "出", "了", "一个", "叛徒"]);
//! ```
//!
//! ```rust
//! # #[cfg(feature = "tfidf")] {
//! use jieba_rs_yada::Jieba;
//! use jieba_rs_yada::{TfIdf, KeywordExtract};
//!
//! fn main() {
//!     let jieba = Jieba::new();
//!     let keyword_extractor = TfIdf::default();
//!     let top_k = keyword_extractor.extract_keywords(
//!         &jieba,
//!         "今天纽约的天气真好啊，京华大酒店的张尧经理吃了一只北京烤鸭。后天纽约的天气不好，昨天纽约的天气也不好，北京烤鸭真好吃",
//!         3,
//!         vec![],
//!     );
//!     println!("{:?}", top_k);
//! }
//! # }
//! ```
//!
//! ```rust
//! # #[cfg(feature = "textrank")] {
//! use jieba_rs_yada::Jieba;
//! use jieba_rs_yada::{TextRank, KeywordExtract};
//!
//! fn main() {
//!     let jieba = Jieba::new();
//!     let keyword_extractor = TextRank::default();
//!     let top_k = keyword_extractor.extract_keywords(
//!         &jieba,
//!         "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
//!         6,
//!         vec![String::from("ns"), String::from("n"), String::from("vn"), String::from("v")],
//!     );
//!     println!("{:?}", top_k);
//! }
//! # }
//! ```
//!
//! ## Enabling Additional Features
//!
//! * `default-dict` feature enables embedded dictionary, this features is enabled by default
//! * `tfidf` feature enables TF-IDF keywords extractor
//! * `textrank` feature enables TextRank keywords extractor
//!
//! ```toml
//! [dependencies]
//! jieba-rs-yada = { version = "0.1", features = ["tfidf", "textrank"] }
//! ```
//!

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::io::BufRead;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use regex::{Match, Matches, Regex};
use yada::{DoubleArray, builder::DoubleArrayBuilder};

pub(crate) type FxHashMap<K, V> = HashMap<K, V, rustc_hash::FxBuildHasher>;

pub use crate::errors::Error;
#[cfg(feature = "textrank")]
pub use crate::keywords::textrank::TextRank;
#[cfg(feature = "tfidf")]
pub use crate::keywords::tfidf::TfIdf;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
pub use crate::keywords::{DEFAULT_STOP_WORDS, Keyword, KeywordExtract, KeywordExtractConfig};

mod errors;
mod hmm;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
mod keywords;
mod sparse_dag;

#[cfg(feature = "default-dict")]
include_flate::flate!(static DEFAULT_DICT: str from "src/data/dict.txt");

use sparse_dag::StaticSparseDAG;

thread_local! {
    static RE_HAN_DEFAULT: Regex = Regex::new(r"([\u{3400}-\u{4DBF}\u{4E00}-\u{9FFF}\u{F900}-\u{FAFF}\u{20000}-\u{2A6DF}\u{2A700}-\u{2B73F}\u{2B740}-\u{2B81F}\u{2B820}-\u{2CEAF}\u{2CEB0}-\u{2EBEF}\u{2F800}-\u{2FA1F}a-zA-Z0-9+#&\._%\-]+)").unwrap();
    static RE_SKIP_DEFAULT: Regex = Regex::new(r"(\r\n|\s)").unwrap();
    static RE_HAN_CUT_ALL: Regex = Regex::new(r"([\u{3400}-\u{4DBF}\u{4E00}-\u{9FFF}\u{F900}-\u{FAFF}\u{20000}-\u{2A6DF}\u{2A700}-\u{2B73F}\u{2B740}-\u{2B81F}\u{2B820}-\u{2CEAF}\u{2CEB0}-\u{2EBEF}\u{2F800}-\u{2FA1F}]+)").unwrap();
    static RE_SKIP_CUT_ALL: Regex = Regex::new(r"[^a-zA-Z0-9+#\n]").unwrap();
    static HMM_CONTEXT: std::cell::RefCell<hmm::HmmContext> = std::cell::RefCell::new(hmm::HmmContext::default());
}

struct SplitMatches<'r, 't> {
    finder: Matches<'r, 't>,
    text: &'t str,
    last: usize,
    matched: Option<Match<'t>>,
}

impl<'r, 't> SplitMatches<'r, 't> {
    #[inline]
    fn new(re: &'r Regex, text: &'t str) -> SplitMatches<'r, 't> {
        SplitMatches {
            finder: re.find_iter(text),
            text,
            last: 0,
            matched: None,
        }
    }
}

#[derive(Debug)]
pub(crate) enum SplitState<'t> {
    Unmatched(&'t str),
    Matched(Match<'t>),
}

impl<'t> SplitState<'t> {
    #[inline]
    fn as_str(&self) -> &'t str {
        match self {
            SplitState::Unmatched(t) => t,
            SplitState::Matched(matched) => matched.as_str(),
        }
    }

    #[inline]
    pub fn is_matched(&self) -> bool {
        matches!(self, SplitState::Matched(_))
    }
}

impl<'t> Iterator for SplitMatches<'_, 't> {
    type Item = SplitState<'t>;

    fn next(&mut self) -> Option<SplitState<'t>> {
        if let Some(matched) = self.matched.take() {
            return Some(SplitState::Matched(matched));
        }
        match self.finder.next() {
            None => {
                if self.last >= self.text.len() {
                    None
                } else {
                    let s = &self.text[self.last..];
                    self.last = self.text.len();
                    Some(SplitState::Unmatched(s))
                }
            }
            Some(m) => {
                if self.last == m.start() {
                    self.last = m.end();
                    Some(SplitState::Matched(m))
                } else {
                    let unmatched = &self.text[self.last..m.start()];
                    self.last = m.end();
                    self.matched = Some(m);
                    Some(SplitState::Unmatched(unmatched))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizeMode {
    /// Default mode
    Default,
    /// Search mode
    Search,
}

/// A Token
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token<'a> {
    /// Word of the token
    pub word: &'a str,
    /// Unicode start position of the token
    pub start: usize,
    /// Unicode end position of the token
    pub end: usize,
}

/// A tagged word
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tag<'a> {
    /// Word
    pub word: &'a str,
    /// Word tag
    pub tag: &'a str,
}

#[derive(Debug, Clone)]
struct Record {
    freq: usize,
    tag: Box<str>,
    word: String,
}

impl Record {
    #[inline(always)]
    fn new(freq: usize, tag: Box<str>, word: String) -> Self {
        Self { freq, tag, word }
    }
}

/// Backing storage for dictionary records.
///
/// Supports two modes:
/// - `Owned`: records live in heap-allocated `Vec<Record>` (used during construction via `new()`)
/// - `Mapped`: records are zero-copy references into a memory-mapped cache file, shared across
///   processes through the OS page cache
#[derive(Clone)]
enum RecordStore {
    Owned(Vec<Record>),
    Mapped {
        /// Shared mmap handle (same `Arc<Mmap>` as `DaData::Mapped`)
        mmap: Arc<Mmap>,
        /// Number of records
        count: usize,
        /// Byte offset in the mmap where the freq array `[u64; count]` starts
        freq_offset: usize,
        /// Byte offset in the mmap where the tag index `[(u32, u32); count]` starts
        tag_index_offset: usize,
        /// Byte offset in the mmap where the word index `[(u32, u32); count]` starts
        #[allow(dead_code)]
        word_index_offset: usize,
        /// Byte offset in the mmap where the strings blob starts
        strings_offset: usize,
    },
}

impl RecordStore {
    #[allow(dead_code)]
    fn len(&self) -> usize {
        match self {
            RecordStore::Owned(v) => v.len(),
            RecordStore::Mapped { count, .. } => *count,
        }
    }

    #[inline]
    fn freq(&self, word_id: usize) -> usize {
        match self {
            RecordStore::Owned(v) => v[word_id].freq,
            RecordStore::Mapped { mmap, freq_offset, .. } => {
                let offset = *freq_offset + word_id * 8;
                u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap()) as usize
            }
        }
    }

    #[inline]
    fn tag(&self, word_id: usize) -> &str {
        match self {
            RecordStore::Owned(v) => &v[word_id].tag,
            RecordStore::Mapped { mmap, tag_index_offset, strings_offset, .. } => {
                let idx_offset = *tag_index_offset + word_id * 8;
                let str_off = u32::from_le_bytes(mmap[idx_offset..idx_offset + 4].try_into().unwrap()) as usize;
                let str_len = u32::from_le_bytes(mmap[idx_offset + 4..idx_offset + 8].try_into().unwrap()) as usize;
                let abs_offset = *strings_offset + str_off;
                std::str::from_utf8(&mmap[abs_offset..abs_offset + str_len]).unwrap()
            }
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn word(&self, word_id: usize) -> &str {
        match self {
            RecordStore::Owned(v) => &v[word_id].word,
            RecordStore::Mapped { mmap, word_index_offset, strings_offset, .. } => {
                let idx_offset = *word_index_offset + word_id * 8;
                let str_off = u32::from_le_bytes(mmap[idx_offset..idx_offset + 4].try_into().unwrap()) as usize;
                let str_len = u32::from_le_bytes(mmap[idx_offset + 4..idx_offset + 8].try_into().unwrap()) as usize;
                let abs_offset = *strings_offset + str_off;
                std::str::from_utf8(&mmap[abs_offset..abs_offset + str_len]).unwrap()
            }
        }
    }
}

impl fmt::Debug for RecordStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecordStore::Owned(v) => write!(f, "RecordStore::Owned({} records)", v.len()),
            RecordStore::Mapped { count, .. } => write!(f, "RecordStore::Mapped({count} records)"),
        }
    }
}

/// Backing storage for the Double Array Trie data.
///
/// Supports two modes:
/// - `Owned`: data lives in a heap-allocated `Vec<u8>` (used during construction)
/// - `Mapped`: data is memory-mapped from a file via `Arc<Mmap>` (used for multi-process sharing)
#[derive(Clone)]
enum DaData {
    Owned(Vec<u8>),
    Mapped {
        mmap: Arc<Mmap>,
        offset: usize,
        length: usize,
    },
}

impl DaData {
    fn as_slice(&self) -> &[u8] {
        match self {
            DaData::Owned(v) => v.as_slice(),
            DaData::Mapped { mmap, offset, length } => &mmap[*offset..*offset + *length],
        }
    }

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}

impl fmt::Debug for DaData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DaData::Owned(v) => write!(f, "DaData::Owned({} bytes)", v.len()),
            DaData::Mapped { offset, length, .. } => {
                write!(f, "DaData::Mapped(offset={offset}, length={length})")
            }
        }
    }
}

/// Cache file format constants
const CACHE_MAGIC: &[u8; 4] = b"JBDA";
const CACHE_VERSION: u32 = 2;

/// Header v2 layout (little-endian):
///
/// | Field            | Size    | Description                                       |
/// |------------------|---------|---------------------------------------------------|
/// | magic            | 4 bytes | `b"JBDA"`                                         |
/// | version          | 4 bytes | format version (`2`)                              |
/// | total            | 8 bytes | sum of all word frequencies                       |
/// | records_count    | 4 bytes | number of dictionary entries                      |
/// | da_data_offset   | 8 bytes | byte offset where DAT data begins                 |
///
/// After the header, the records region contains (all contiguous, zero-copy friendly):
///
/// | Section          | Size                  | Description                              |
/// |------------------|-----------------------|------------------------------------------|
/// | freq_array       | 8 × N                | `[u64; N]` word frequencies              |
/// | tag_index        | 8 × N                | `[(u32 offset, u32 len); N]` tag refs    |
/// | word_index       | 8 × N                | `[(u32 offset, u32 len); N]` word refs   |
/// | strings_blob     | variable              | raw UTF-8 bytes for all tags and words   |
///
/// Then the DAT data follows at `da_data_offset`.
const CACHE_HEADER_SIZE: usize = 28;

/// Jieba segmentation
#[derive(Clone)]
pub struct Jieba {
    record_store: RecordStore,
    da_data: DaData,
    total: usize,
}

impl fmt::Debug for Jieba {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Jieba")
            .field("record_store", &self.record_store)
            .field("total_freq", &self.total)
            .finish()
    }
}

#[cfg(feature = "default-dict")]
impl Default for Jieba {
    fn default() -> Self {
        Jieba::new()
    }
}

impl Jieba {
    /// Create a new instance with empty dict
    pub fn empty() -> Self {
        Jieba {
            record_store: RecordStore::Owned(Vec::new()),
            da_data: DaData::Owned(Vec::new()),
            total: 0,
        }
    }

    /// Returns a mutable reference to the owned records vector.
    /// Panics if the record store is in Mapped mode.
    fn records_mut(&mut self) -> &mut Vec<Record> {
        match &mut self.record_store {
            RecordStore::Owned(v) => v,
            RecordStore::Mapped { .. } => panic!("cannot mutate records in mmap mode"),
        }
    }

    /// Returns a reference to the owned records vector.
    /// Panics if the record store is in Mapped mode.
    fn records(&self) -> &Vec<Record> {
        match &self.record_store {
            RecordStore::Owned(v) => v,
            RecordStore::Mapped { .. } => panic!("cannot access owned records in mmap mode"),
        }
    }

    /// Create a new instance with embed dict
    ///
    /// Requires `default-dict` feature to be enabled.
    #[cfg(feature = "default-dict")]
    pub fn new() -> Self {
        let mut instance = Self::empty();
        instance.load_default_dict();
        instance
    }

    /// Create a new instance with dict
    pub fn with_dict<R: BufRead>(dict: &mut R) -> Result<Self, Error> {
        let mut instance = Self::empty();
        instance.load_dict(dict)?;
        Ok(instance)
    }

    /// Loads the default dictionary into the instance.
    ///
    /// This method reads the default dictionary from a predefined byte slice (`DEFAULT_DICT`)
    /// and loads it into the current instance using the `load_dict` method.
    ///
    /// # Arguments
    ///
    /// * `&mut self` - Mutable reference to the current instance.
    ///
    /// Requires `default-dict` feature to be enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// use jieba_rs_yada::Jieba;
    ///
    /// let mut instance = Jieba::empty();
    /// instance.load_default_dict(); // Loads the default dictionary into the instance
    /// assert!(instance.has_word("我们"), "The word '我们' should be in the dictionary after loading the default dictionary");
    /// ```
    #[cfg(feature = "default-dict")]
    pub fn load_default_dict(&mut self) {
        use std::io::BufReader;

        let mut default_dict = BufReader::new(DEFAULT_DICT.as_bytes());
        self.load_dict(&mut default_dict).unwrap();
    }

    /// Clears all data
    ///
    /// This method performs the following actions:
    /// 1. Clears the `records` list, removing all entries.
    /// 2. Resets the double-array trie data.
    /// 3. Sets `total` to 0, resetting the count.
    ///
    /// # Arguments
    ///
    /// * `&mut self` - Mutable reference to the current instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use jieba_rs_yada::Jieba;
    ///
    /// let mut instance = Jieba::new();
    /// assert!(instance.has_word("我们"), "The word '我们' should be in the dictionary after loading the default dictionary");
    /// instance.clear(); // clear all dict data
    /// assert!(!instance.has_word("我们"), "The word '我们' should not be in the dictionary after clearing the dictionary");
    /// ```
    pub fn clear(&mut self) {
        self.record_store = RecordStore::Owned(Vec::new());
        self.da_data = DaData::Owned(Vec::new());
        self.total = 0;
    }

    /// Add word to dict, return `freq`
    ///
    /// `freq`: if `None`, will be given by [suggest_freq](#method.suggest_freq)
    ///
    /// `tag`: if `None`, will be given `""`
    pub fn add_word(&mut self, _word: &str, _freq: Option<usize>, _tag: Option<&str>) -> usize {
        unimplemented!("add_word is not supported with static Double Array Trie backend")
    }

    /// Checks if a word exists in the dictionary.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to check.
    ///
    /// # Returns
    ///
    /// * `bool` - Whether the word exists in the dictionary.
    pub fn has_word(&self, word: &str) -> bool {
        if self.da_data.is_empty() {
            return false;
        }
        let da = DoubleArray::new(self.da_data.as_slice());
        da.exact_match_search(word.as_bytes()).is_some()
    }

    /// Loads a dictionary by adding entries to the existing dictionary rather than resetting it.
    ///
    /// This function reads from a `BufRead` source, parsing each line as a dictionary entry. Each entry
    /// is expected to contain a word, its frequency, and optionally a tag.
    ///
    /// # Type Parameters
    ///
    /// * `R`: A type that implements the `BufRead` trait, used for reading lines from the dictionary.
    ///
    /// # Arguments
    ///
    /// * `dict` - A mutable reference to a `BufRead` source containing the dictionary entries.
    ///
    /// # Returns
    ///
    /// * `Result<(), Error>` - Returns `Ok(())` if the dictionary is successfully loaded; otherwise,
    ///   returns an error describing what went wrong.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * There is an issue reading from the provided `BufRead` source.
    /// * A line in the dictionary file contains invalid frequency data (not a valid integer).
    pub fn load_dict<R: BufRead>(&mut self, dict: &mut R) -> Result<(), Error> {
        let mut buf = String::new();
        self.total = 0;

        // Temporary index for O(1) dedup during loading
        let records = self.records_mut();
        let mut word_to_id: FxHashMap<String, u32> = records
            .iter()
            .enumerate()
            .map(|(id, r)| (r.word.clone(), id as u32))
            .collect();

        let mut line_no = 0;
        while dict.read_line(&mut buf)? > 0 {
            {
                line_no += 1;
                let mut iter = buf.split_whitespace();
                if let Some(word) = iter.next() {
                    let freq = iter
                        .next()
                        .map(|x| {
                            x.parse::<usize>().map_err(|e| {
                                Error::InvalidDictEntry(format!(
                                    "line {line_no} `{buf}` frequency {x} is not a valid integer: {e}"
                                ))
                            })
                        })
                        .unwrap_or(Ok(0))?;
                    let tag = iter.next().unwrap_or("");

                    if let Some(&word_id) = word_to_id.get(word) {
                        records[word_id as usize].freq = freq;
                    } else {
                        let word_id = records.len() as u32;
                        records.push(Record::new(freq, tag.into(), word.to_string()));
                        word_to_id.insert(word.to_string(), word_id);
                    }
                }
            }
            buf.clear();
        }
        self.total = self.records().iter().map(|n| n.freq).sum();

        // Build DAT from records
        let mut keyset: Vec<(&[u8], u32)> = self
            .records()
            .iter()
            .enumerate()
            .map(|(id, r)| (r.word.as_bytes(), id as u32))
            .collect();
        keyset.sort_by(|a, b| a.0.cmp(b.0));
        self.da_data = if keyset.is_empty() {
            DaData::Owned(Vec::new())
        } else {
            DaData::Owned(DoubleArrayBuilder::build(&keyset).expect("failed to build DoubleArray"))
        };
        Ok(())
    }

    /// Save the dictionary cache to a file for later mmap-based loading.
    ///
    /// The file is written atomically using a write-to-temp-then-rename strategy,
    /// which is safe for concurrent multi-process access: readers will either see
    /// the complete old file or the complete new file, never a partially-written one.
    ///
    /// ## File format (little-endian)
    ///
    /// ## Example
    ///
    /// ```no_run
    /// use jieba_rs_yada::Jieba;
    ///
    /// let jieba = Jieba::new();
    /// jieba.save_cache("/tmp/jieba.dict.cache").unwrap();
    /// ```
    pub fn save_cache<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        use std::io::Write;

        let path = path.as_ref();
        let tmp_path = path.with_extension(format!("tmp.{}", std::process::id()));

        let records = self.records();
        let da_slice = self.da_data.as_slice();

        // Build the strings blob and index tables for tags and words
        let record_count = records.len();
        let mut strings_blob: Vec<u8> = Vec::new();
        let mut tag_index: Vec<(u32, u32)> = Vec::with_capacity(record_count);
        let mut word_index: Vec<(u32, u32)> = Vec::with_capacity(record_count);

        for record in records {
            let tag_offset = strings_blob.len() as u32;
            strings_blob.extend_from_slice(record.tag.as_bytes());
            let tag_len = record.tag.len() as u32;
            tag_index.push((tag_offset, tag_len));

            let word_offset = strings_blob.len() as u32;
            strings_blob.extend_from_slice(record.word.as_bytes());
            let word_len = record.word.len() as u32;
            word_index.push((word_offset, word_len));
        }

        // Records region: freq_array + tag_index + word_index + strings_blob
        let freq_array_size = record_count * 8;
        let tag_index_size = record_count * 8;
        let word_index_size = record_count * 8;
        let records_size = freq_array_size + tag_index_size + word_index_size + strings_blob.len();
        let da_data_offset = (CACHE_HEADER_SIZE + records_size) as u64;

        // Write to temporary file
        let mut file = std::fs::File::create(&tmp_path).map_err(|e| {
            Error::InvalidDictEntry(format!("failed to create cache file {}: {e}", tmp_path.display()))
        })?;

        let write_err = |e: std::io::Error, context: &str| -> Error {
            Error::InvalidDictEntry(format!("failed to write {context}: {e}"))
        };

        // Header
        file.write_all(CACHE_MAGIC).map_err(|e| write_err(e, "cache magic"))?;
        file.write_all(&CACHE_VERSION.to_le_bytes()).map_err(|e| write_err(e, "cache version"))?;
        file.write_all(&(self.total as u64).to_le_bytes()).map_err(|e| write_err(e, "total"))?;
        file.write_all(&(record_count as u32).to_le_bytes())
            .map_err(|e| write_err(e, "records count"))?;
        file.write_all(&da_data_offset.to_le_bytes()).map_err(|e| write_err(e, "da_data_offset"))?;

        // freq_array: [u64; N]
        for record in records {
            file.write_all(&(record.freq as u64).to_le_bytes())
                .map_err(|e| write_err(e, "record freq"))?;
        }

        // tag_index: [(u32 offset, u32 len); N]
        for &(offset, len) in &tag_index {
            file.write_all(&offset.to_le_bytes()).map_err(|e| write_err(e, "tag offset"))?;
            file.write_all(&len.to_le_bytes()).map_err(|e| write_err(e, "tag len"))?;
        }

        // word_index: [(u32 offset, u32 len); N]
        for &(offset, len) in &word_index {
            file.write_all(&offset.to_le_bytes()).map_err(|e| write_err(e, "word offset"))?;
            file.write_all(&len.to_le_bytes()).map_err(|e| write_err(e, "word len"))?;
        }

        // strings_blob
        file.write_all(&strings_blob).map_err(|e| write_err(e, "strings blob"))?;

        // DAT data
        file.write_all(da_slice).map_err(|e| write_err(e, "DAT data"))?;

        file.flush().map_err(|e| write_err(e, "flush"))?;
        drop(file);

        // Atomic rename: safe for concurrent readers
        std::fs::rename(&tmp_path, path).map_err(|e| {
            Error::InvalidDictEntry(format!(
                "failed to rename {} -> {}: {e}",
                tmp_path.display(),
                path.display()
            ))
        })?;

        Ok(())
    }

    /// Load a Jieba instance from a previously saved cache file using mmap.
    ///
    /// The DAT portion of the file is memory-mapped, so multiple processes that
    /// load the same cache file will **share the same physical memory pages**
    /// through the OS page cache, significantly reducing memory usage.
    ///
    /// ## Multi-process safety
    ///
    /// If the cache file does not yet exist, this method will:
    /// 1. Acquire an exclusive file lock on a `.lock` sidecar file.
    /// 2. Double-check whether another process has created the cache in the meantime.
    /// 3. If not, build the dictionary from the default dict and write the cache.
    /// 4. Release the lock so waiting processes can proceed.
    ///
    /// Processes that arrive while the lock is held will block until the cache is
    /// ready, then mmap it directly.
    ///
    /// ## Example
    ///
    /// ```no_run
    /// use jieba_rs_yada::Jieba;
    ///
    /// let jieba = Jieba::load_from_cache("/tmp/jieba.dict.cache").unwrap();
    /// let words = jieba.cut("测试分词", false);
    /// ```
    pub fn load_from_cache<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::load_or_build_cache(path, || Ok(Self::new()))
    }

    /// Load a Jieba instance from a cache file that merges default + user dict.
    ///
    /// - If `cache_path` already exists, mmap-load it directly (no lock needed).
    /// - Otherwise, acquire flock on `{cache_path}.lock`, double-check, and if
    ///   still missing: build from default dict + `user_dict_content`, save
    ///   cache atomically, then mmap-load.
    ///
    /// `user_dict_content` is the raw text of the user dictionary, one entry
    /// per line in `word freq [tag]` format (same as `load_dict` input).
    ///
    /// The caller is responsible for:
    /// - Computing the content hash and embedding it in the cache filename
    ///   (so different dict content -> different cache file).
    /// - Detecting when the user dict changes and calling this method with
    ///   the new cache path.
    pub fn load_from_cache_with_user_dict<P: AsRef<Path>>(
        cache_path: P,
        user_dict_content: &str,
    ) -> Result<Self, Error> {
        Self::load_or_build_cache(cache_path, || {
            let mut instance = Self::new();
            instance.load_dict(&mut std::io::BufReader::new(user_dict_content.as_bytes()))?;
            Ok(instance)
        })
    }

    fn load_or_build_cache<P, F>(cache_path: P, build_fn: F) -> Result<Self, Error>
    where
        P: AsRef<Path>,
        F: FnOnce() -> Result<Self, Error>,
    {
        let path = cache_path.as_ref();

        // Fast path: cache already exists
        if path.exists() {
            return Self::mmap_load(path);
        }

        // Slow path: coordinate with other processes via file lock
        let lock_path = path.with_extension("lock");
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| {
                Error::InvalidDictEntry(format!("failed to open lock file {}: {e}", lock_path.display()))
            })?;

        Self::flock_exclusive(&lock_file)?;

        // Double-check: another process may have created the cache while we waited
        let result = if path.exists() {
            Self::mmap_load(path)
        } else {
            let instance = build_fn()?;
            instance.save_cache(path)?;
            Self::mmap_load(path)
        };

        Self::flock_unlock(&lock_file)?;
        drop(lock_file);

        result
    }

    /// Memory-map a cache file and construct a Jieba instance from it.
    ///
    /// All data (freq, tag, word, DAT) remains in the mmap region and is
    /// referenced zero-copy. Multiple processes mapping the same file share
    /// the same physical memory pages through the OS page cache.
    fn mmap_load(path: &Path) -> Result<Self, Error> {
        let file = std::fs::File::open(path).map_err(|e| {
            Error::InvalidDictEntry(format!("failed to open cache file {}: {e}", path.display()))
        })?;

        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            Error::InvalidDictEntry(format!("failed to mmap cache file {}: {e}", path.display()))
        })?;

        let data = &mmap[..];
        if data.len() < CACHE_HEADER_SIZE {
            return Err(Error::InvalidDictEntry("cache file too small".into()));
        }

        // Parse header
        if &data[0..4] != CACHE_MAGIC {
            return Err(Error::InvalidDictEntry("invalid cache magic".into()));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != CACHE_VERSION {
            return Err(Error::InvalidDictEntry(format!(
                "unsupported cache version {version}, expected {CACHE_VERSION}"
            )));
        }
        let total = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        let records_count = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let da_data_offset = u64::from_le_bytes(data[20..28].try_into().unwrap()) as usize;

        if da_data_offset > data.len() {
            return Err(Error::InvalidDictEntry("da_data_offset exceeds file size".into()));
        }

        // Validate records region layout:
        //   freq_array:  [u64; N]           at CACHE_HEADER_SIZE
        //   tag_index:   [(u32, u32); N]    at CACHE_HEADER_SIZE + 8*N
        //   word_index:  [(u32, u32); N]    at CACHE_HEADER_SIZE + 16*N
        //   strings_blob:                   at CACHE_HEADER_SIZE + 24*N
        let freq_offset = CACHE_HEADER_SIZE;
        let tag_index_offset = freq_offset + records_count * 8;
        let word_index_offset = tag_index_offset + records_count * 8;
        let strings_offset = word_index_offset + records_count * 8;

        if strings_offset > da_data_offset {
            return Err(Error::InvalidDictEntry("records index tables exceed da_data_offset".into()));
        }

        // Validate that strings referenced by tag/word indices are within bounds
        let strings_region_len = da_data_offset - strings_offset;
        for i in 0..records_count {
            // Validate tag index entry
            let ti_off = tag_index_offset + i * 8;
            let tag_str_off = u32::from_le_bytes(data[ti_off..ti_off + 4].try_into().unwrap()) as usize;
            let tag_str_len = u32::from_le_bytes(data[ti_off + 4..ti_off + 8].try_into().unwrap()) as usize;
            if tag_str_off + tag_str_len > strings_region_len {
                return Err(Error::InvalidDictEntry(format!(
                    "tag string for record {i} exceeds strings region"
                )));
            }

            // Validate word index entry
            let wi_off = word_index_offset + i * 8;
            let word_str_off = u32::from_le_bytes(data[wi_off..wi_off + 4].try_into().unwrap()) as usize;
            let word_str_len = u32::from_le_bytes(data[wi_off + 4..wi_off + 8].try_into().unwrap()) as usize;
            if word_str_off + word_str_len > strings_region_len {
                return Err(Error::InvalidDictEntry(format!(
                    "word string for record {i} exceeds strings region"
                )));
            }
        }

        let da_data_length = data.len() - da_data_offset;
        let mmap_arc = Arc::new(mmap);

        Ok(Jieba {
            record_store: RecordStore::Mapped {
                mmap: Arc::clone(&mmap_arc),
                count: records_count,
                freq_offset,
                tag_index_offset,
                word_index_offset,
                strings_offset,
            },
            da_data: DaData::Mapped {
                mmap: mmap_arc,
                offset: da_data_offset,
                length: da_data_length,
            },
            total,
        })
    }

    #[cfg(unix)]
    fn flock_exclusive(file: &std::fs::File) -> Result<(), Error> {
        use std::os::unix::io::AsRawFd;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        if result != 0 {
            return Err(Error::InvalidDictEntry(format!(
                "flock(LOCK_EX) failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(())
    }

    #[cfg(unix)]
    fn flock_unlock(file: &std::fs::File) -> Result<(), Error> {
        use std::os::unix::io::AsRawFd;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
        if result != 0 {
            return Err(Error::InvalidDictEntry(format!(
                "flock(LOCK_UN) failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(())
    }

    #[cfg(not(unix))]
    fn flock_exclusive(_file: &std::fs::File) -> Result<(), Error> {
        Ok(())
    }

    #[cfg(not(unix))]
    fn flock_unlock(_file: &std::fs::File) -> Result<(), Error> {
        Ok(())
    }

    fn get_word_freq(&self, word: &str, default: usize) -> usize {
        if self.da_data.is_empty() {
            return default;
        }
        let da = DoubleArray::new(self.da_data.as_slice());
        match da.exact_match_search(word.as_bytes()) {
            Some(word_id) => self.record_store.freq(word_id as usize),
            None => default,
        }
    }

    /// Suggest word frequency to force the characters in a word to be joined or split.
    pub fn suggest_freq(&self, segment: &str) -> usize {
        let logtotal = (self.total as f64).ln();
        let logfreq = self.cut(segment, false).iter().fold(0f64, |freq, word| {
            freq + (self.get_word_freq(word, 1) as f64).ln() - logtotal
        });
        std::cmp::max((logfreq + logtotal).exp() as usize + 1, self.get_word_freq(segment, 1))
    }

    #[allow(clippy::ptr_arg)]
    fn calc(&self, sentence: &str, dag: &StaticSparseDAG, route: &mut Vec<(f64, usize)>) {
        let str_len = sentence.len();

        if str_len + 1 > route.len() {
            route.resize(str_len + 1, (0.0, 0));
        }

        let da = DoubleArray::new(self.da_data.as_slice());
        let logtotal = (self.total as f64).ln();
        let mut prev_byte_start = str_len;
        let curr = sentence.char_indices().map(|x| x.0).rev();
        for byte_start in curr {
            let pair = dag
                .iter_edges(byte_start)
                .map(|byte_end| {
                    let wfrag = &sentence[byte_start..byte_end];

                    let freq = if let Some(word_id) = da.exact_match_search(wfrag.as_bytes()) {
                        self.record_store.freq(word_id as usize)
                    } else {
                        1
                    };

                    ((freq as f64).ln() - logtotal + route[byte_end].0, byte_end)
                })
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));

            if let Some(p) = pair {
                route[byte_start] = p;
            } else {
                let byte_end = prev_byte_start;
                let freq = 1;
                route[byte_start] = ((freq as f64).ln() - logtotal + route[byte_end].0, byte_end);
            }

            prev_byte_start = byte_start;
        }
    }

    fn dag(&self, sentence: &str, dag: &mut StaticSparseDAG) {
        let da = DoubleArray::new(self.da_data.as_slice());
        for (byte_start, _) in sentence.char_indices().peekable() {
            dag.start(byte_start);
            let haystack = &sentence[byte_start..];

            for (_, end_index) in da.common_prefix_search(haystack.as_bytes()) {
                dag.insert(end_index + byte_start);
            }

            dag.commit();
        }
    }

    fn cut_all_internal<'a>(&self, sentence: &'a str, words: &mut Vec<&'a str>) {
        let str_len = sentence.len();
        let mut dag = StaticSparseDAG::with_size_hint(sentence.len());
        self.dag(sentence, &mut dag);

        let curr = sentence.char_indices().map(|x| x.0);
        for byte_start in curr {
            for byte_end in dag.iter_edges(byte_start) {
                let word = if byte_end == str_len {
                    &sentence[byte_start..]
                } else {
                    &sentence[byte_start..byte_end]
                };

                words.push(word)
            }
        }
    }

    fn cut_dag_no_hmm<'a>(
        &self,
        sentence: &'a str,
        words: &mut Vec<&'a str>,
        route: &mut Vec<(f64, usize)>,
        dag: &mut StaticSparseDAG,
    ) {
        self.dag(sentence, dag);
        self.calc(sentence, dag, route);
        let mut x = 0;
        let mut left: Option<usize> = None;

        while x < sentence.len() {
            let y = route[x].1;
            let l_str = &sentence[x..y];

            if l_str.chars().count() == 1 && l_str.chars().all(|ch| ch.is_ascii_alphanumeric()) {
                if left.is_none() {
                    left = Some(x);
                }
            } else {
                if let Some(byte_start) = left {
                    let word = &sentence[byte_start..x];
                    words.push(word);
                    left = None;
                }

                words.push(l_str);
            }
            x = y;
        }

        if let Some(byte_start) = left {
            let word = &sentence[byte_start..];
            words.push(word);
        }

        dag.clear();
        route.clear();
    }

    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn cut_dag_hmm<'a>(
        &self,
        sentence: &'a str,
        words: &mut Vec<&'a str>,
        route: &mut Vec<(f64, usize)>,
        dag: &mut StaticSparseDAG,
        hmm_context: &mut hmm::HmmContext,
    ) {
        self.dag(sentence, dag);
        self.calc(sentence, dag, route);
        let mut x = 0;
        let mut left: Option<usize> = None;

        while x < sentence.len() {
            let y = route[x].1;

            if sentence[x..y].chars().count() == 1 {
                if left.is_none() {
                    left = Some(x);
                }
            } else {
                if let Some(byte_start) = left {
                    let byte_end = x;
                    let word = &sentence[byte_start..byte_end];
                    if word.chars().count() == 1 {
                        words.push(word);
                    } else if !self.has_word(word) {
                        hmm::cut_with_allocated_memory(word, words, hmm_context);
                    } else {
                        let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                        while let Some(byte_start) = word_indices.next() {
                            if let Some(byte_end) = word_indices.peek() {
                                words.push(&word[byte_start..*byte_end]);
                            } else {
                                words.push(&word[byte_start..]);
                            }
                        }
                    }
                    left = None;
                }
                let word = &sentence[x..y];
                words.push(word);
            }
            x = y;
        }

        if let Some(byte_start) = left {
            let word = &sentence[byte_start..];

            if word.chars().count() == 1 {
                words.push(word);
            } else if !self.has_word(word) {
                hmm::cut(word, words);
            } else {
                let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                while let Some(byte_start) = word_indices.next() {
                    if let Some(byte_end) = word_indices.peek() {
                        words.push(&word[byte_start..*byte_end]);
                    } else {
                        words.push(&word[byte_start..]);
                    }
                }
            }
        }

        dag.clear();
        route.clear();
    }

    #[allow(non_snake_case)]
    fn cut_internal<'a>(&self, sentence: &'a str, cut_all: bool, hmm: bool) -> Vec<&'a str> {
        let re_han = if cut_all { &RE_HAN_CUT_ALL } else { &RE_HAN_DEFAULT };
        let re_skip = if cut_all { &RE_SKIP_CUT_ALL } else { &RE_SKIP_DEFAULT };

        re_han.with(|re_han| {
            re_skip.with(|re_skip| {
                let heuristic_capacity = sentence.len() / 2;
                let mut words = Vec::with_capacity(heuristic_capacity);

                let splitter = SplitMatches::new(re_han, sentence);
                let mut route = Vec::with_capacity(heuristic_capacity);
                let mut dag = StaticSparseDAG::with_size_hint(heuristic_capacity);

                for state in splitter {
                    match state {
                        SplitState::Matched(_) => {
                            let block = state.as_str();
                            assert!(!block.is_empty());

                            if cut_all {
                                self.cut_all_internal(block, &mut words);
                            } else if hmm {
                                HMM_CONTEXT.with(|ctx| {
                                    let mut hmm_context = ctx.borrow_mut();
                                    self.cut_dag_hmm(block, &mut words, &mut route, &mut dag, &mut hmm_context);
                                });
                            } else {
                                self.cut_dag_no_hmm(block, &mut words, &mut route, &mut dag);
                            }
                        }
                        SplitState::Unmatched(_) => {
                            let block = state.as_str();
                            assert!(!block.is_empty());

                            let skip_splitter = SplitMatches::new(re_skip, block);
                            for skip_state in skip_splitter {
                                let word = skip_state.as_str();
                                if word.is_empty() {
                                    continue;
                                }
                                if cut_all || skip_state.is_matched() {
                                    words.push(word);
                                } else {
                                    let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                                    while let Some(byte_start) = word_indices.next() {
                                        if let Some(byte_end) = word_indices.peek() {
                                            words.push(&word[byte_start..*byte_end]);
                                        } else {
                                            words.push(&word[byte_start..]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                words
            })
        })
    }

    /// Cut the input text
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn cut<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<&'a str> {
        self.cut_internal(sentence, false, hmm)
    }

    /// Cut the input text, return all possible words
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    pub fn cut_all<'a>(&self, sentence: &'a str) -> Vec<&'a str> {
        self.cut_internal(sentence, true, false)
    }

    /// Cut the input text in search mode
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn cut_for_search<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<&'a str> {
        let words = self.cut(sentence, hmm);
        let mut new_words = Vec::with_capacity(words.len());
        for word in words {
            let char_indices: Vec<usize> = word.char_indices().map(|x| x.0).collect();
            let char_count = char_indices.len();
            if char_count > 2 {
                for i in 0..char_count - 1 {
                    let byte_start = char_indices[i];
                    let gram2 = if i + 2 < char_count {
                        &word[byte_start..char_indices[i + 2]]
                    } else {
                        &word[byte_start..]
                    };
                    if self.has_word(gram2) {
                        new_words.push(gram2);
                    }
                }
            }
            if char_count > 3 {
                for i in 0..char_count - 2 {
                    let byte_start = char_indices[i];
                    let gram3 = if i + 3 < char_count {
                        &word[byte_start..char_indices[i + 3]]
                    } else {
                        &word[byte_start..]
                    };
                    if self.has_word(gram3) {
                        new_words.push(gram3);
                    }
                }
            }
            new_words.push(word);
        }
        new_words
    }

    /// Tokenize
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `mode`: tokenize mode
    ///
    /// `hmm`: enable HMM or not
    pub fn tokenize<'a>(&self, sentence: &'a str, mode: TokenizeMode, hmm: bool) -> Vec<Token<'a>> {
        let words = self.cut(sentence, hmm);
        let mut tokens = Vec::with_capacity(words.len());
        let mut start = 0;
        match mode {
            TokenizeMode::Default => {
                for word in words {
                    let width = word.chars().count();
                    tokens.push(Token {
                        word,
                        start,
                        end: start + width,
                    });
                    start += width;
                }
            }
            TokenizeMode::Search => {
                for word in words {
                    let width = word.chars().count();
                    if width > 2 {
                        let char_indices: Vec<usize> = word.char_indices().map(|x| x.0).collect();
                        for i in 0..width - 1 {
                            let byte_start = char_indices[i];
                            let gram2 = if i + 2 < width {
                                &word[byte_start..char_indices[i + 2]]
                            } else {
                                &word[byte_start..]
                            };
                            if self.has_word(gram2) {
                                tokens.push(Token {
                                    word: gram2,
                                    start: start + i,
                                    end: start + i + 2,
                                });
                            }
                        }
                        if width > 3 {
                            for i in 0..width - 2 {
                                let byte_start = char_indices[i];
                                let gram3 = if i + 3 < width {
                                    &word[byte_start..char_indices[i + 3]]
                                } else {
                                    &word[byte_start..]
                                };
                                if self.has_word(gram3) {
                                    tokens.push(Token {
                                        word: gram3,
                                        start: start + i,
                                        end: start + i + 3,
                                    });
                                }
                            }
                        }
                    }
                    tokens.push(Token {
                        word,
                        start,
                        end: start + width,
                    });
                    start += width;
                }
            }
        }
        tokens
    }

    /// Tag the input text
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn tag<'a>(&'a self, sentence: &'a str, hmm: bool) -> Vec<Tag<'a>> {
        let words = self.cut(sentence, hmm);
        words
            .into_iter()
            .map(|word| {
                if !self.da_data.is_empty() {
                    let da = DoubleArray::new(self.da_data.as_slice());
                    if let Some(word_id) = da.exact_match_search(word.as_bytes()) {
                        let t = self.record_store.tag(word_id as usize);
                        return Tag { word, tag: t };
                    }
                }
                let mut eng = 0;
                let mut m = 0;
                for chr in word.chars() {
                    if chr.is_ascii_alphanumeric() {
                        eng += 1;
                        if chr.is_ascii_digit() {
                            m += 1;
                        }
                    }
                }
                let tag = if eng == 0 {
                    "x"
                } else if eng == m {
                    "m"
                } else {
                    "eng"
                };
                Tag { word, tag }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Jieba, RE_HAN_DEFAULT, SplitMatches, SplitState, Tag, Token, TokenizeMode};
    use std::io::BufReader;

    #[test]
    fn test_init_with_default_dict() {
        let _ = Jieba::new();
    }

    #[test]
    fn test_has_word() {
        let jieba = Jieba::new();
        assert!(jieba.has_word("中国"));
        assert!(jieba.has_word("开源"));
        assert!(!jieba.has_word("不存在的词"));
    }

    #[test]
    fn test_split_matches() {
        RE_HAN_DEFAULT.with(|re_han| {
            let splitter = SplitMatches::new(
                re_han,
                "👪 PS: 我觉得开源有一个好处，就是能够敦促自己不断改进 👪，避免敞帚自珍",
            );
            for state in splitter {
                match state {
                    SplitState::Matched(_) => {
                        let block = state.as_str();
                        assert!(!block.is_empty());
                    }
                    SplitState::Unmatched(_) => {
                        let block = state.as_str();
                        assert!(!block.is_empty());
                    }
                }
            }
        });
    }

    #[test]
    fn test_split_matches_against_unicode_sip() {
        RE_HAN_DEFAULT.with(|re_han| {
            let splitter = SplitMatches::new(re_han, "讥䶯䶰䶱䶲䶳䶴䶵𦡦");

            let result: Vec<&str> = splitter.map(|x| x.as_str()).collect();
            assert_eq!(result, vec!["讥䶯䶰䶱䶲䶳䶴䶵𦡦"]);
        });
    }

    #[test]
    fn test_cut_all() {
        let jieba = Jieba::new();
        let words = jieba.cut_all("abc网球拍卖会def");
        assert_eq!(
            words,
            vec![
                "abc",
                "网",
                "网球",
                "网球拍",
                "球",
                "球拍",
                "拍",
                "拍卖",
                "拍卖会",
                "卖",
                "会",
                "def"
            ]
        );

        // The cut_all from the python de-facto implementation is loosely defined,
        // And the answer "我, 来到, 北京, 清华, 清华大学, 华大, 大学" from the python implementation looks weird since it drops the single character word even though it is part of the DAG candidates.
        // For example, it includes "华大" but it doesn't include "清" and "学"
        let words = jieba.cut_all("我来到北京清华大学");
        assert_eq!(
            words,
            vec![
                "我",
                "来",
                "来到",
                "到",
                "北",
                "北京",
                "京",
                "清",
                "清华",
                "清华大学",
                "华",
                "华大",
                "大",
                "大学",
                "学"
            ]
        );
    }

    #[test]
    fn test_cut_no_hmm() {
        let jieba = Jieba::new();
        let words = jieba.cut("abc网球拍卖会def", false);
        assert_eq!(words, vec!["abc", "网球", "拍卖会", "def"]);
    }

    #[test]
    fn test_cut_no_hmm1() {
        let jieba = Jieba::new();
        let words = jieba.cut("abc网球拍卖会def！！？\r\n\t", false);
        assert_eq!(
            words,
            vec!["abc", "网球", "拍卖会", "def", "！", "！", "？", "\r\n", "\t"]
        );
    }

    #[test]
    fn test_cut_with_hmm() {
        let jieba = Jieba::new();
        let words = jieba.cut("我们中出了一个叛徒", false);
        assert_eq!(words, vec!["我们", "中", "出", "了", "一个", "叛徒"]);
        let words = jieba.cut("我们中出了一个叛徒", true);
        assert_eq!(words, vec!["我们", "中出", "了", "一个", "叛徒"]);
        let words = jieba.cut("我们中出了一个叛徒👪", true);
        assert_eq!(words, vec!["我们", "中出", "了", "一个", "叛徒", "👪"]);

        let words = jieba.cut("我来到北京清华大学", true);
        assert_eq!(words, vec!["我", "来到", "北京", "清华大学"]);

        let words = jieba.cut("他来到了网易杭研大厦", true);
        assert_eq!(words, vec!["他", "来到", "了", "网易", "杭研", "大厦"]);
    }

    #[test]
    fn test_cut_weicheng() {
        static WEICHENG_TXT: &str = include_str!("../../examples/weicheng/src/weicheng.txt");
        let jieba = Jieba::new();
        for line in WEICHENG_TXT.split('\n') {
            let _ = jieba.cut(line, true);
        }
    }

    #[test]
    fn test_cut_for_search() {
        let jieba = Jieba::new();
        let words = jieba.cut_for_search("南京市长江大桥", true);
        assert_eq!(words, vec!["南京", "京市", "南京市", "长江", "大桥", "长江大桥"]);

        let words = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", true);

        // The python implementation silently filtered "，". but we include it here in the output
        // to let the library user to decide their own filtering strategy
        assert_eq!(
            words,
            vec![
                "小明",
                "硕士",
                "毕业",
                "于",
                "中国",
                "科学",
                "学院",
                "科学院",
                "中国科学院",
                "计算",
                "计算所",
                "，",
                "后",
                "在",
                "日本",
                "京都",
                "大学",
                "日本京都大学",
                "深造"
            ]
        );
    }

    #[test]
    fn test_tag() {
        let jieba = Jieba::new();
        let tags = jieba.tag(
            "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
            true,
        );
        assert_eq!(
            tags,
            vec![
                Tag { word: "我", tag: "r" },
                Tag { word: "是", tag: "v" },
                Tag {
                    word: "拖拉机",
                    tag: "n"
                },
                Tag {
                    word: "学院", tag: "n"
                },
                Tag {
                    word: "手扶拖拉机",
                    tag: "n"
                },
                Tag {
                    word: "专业", tag: "n"
                },
                Tag { word: "的", tag: "uj" },
                Tag { word: "。", tag: "x" },
                Tag {
                    word: "不用", tag: "v"
                },
                Tag {
                    word: "多久", tag: "m"
                },
                Tag { word: "，", tag: "x" },
                Tag { word: "我", tag: "r" },
                Tag { word: "就", tag: "d" },
                Tag { word: "会", tag: "v" },
                Tag {
                    word: "升职", tag: "v"
                },
                Tag {
                    word: "加薪",
                    tag: "nr"
                },
                Tag { word: "，", tag: "x" },
                Tag {
                    word: "当上", tag: "t"
                },
                Tag {
                    word: "CEO",
                    tag: "eng"
                },
                Tag { word: "，", tag: "x" },
                Tag {
                    word: "走上", tag: "v"
                },
                Tag {
                    word: "人生", tag: "n"
                },
                Tag {
                    word: "巅峰", tag: "n"
                },
                Tag { word: "。", tag: "x" }
            ]
        );

        let tags = jieba.tag("今天纽约的天气真好啊，京华大酒店的张尧经理吃了一只北京烤鸭。", true);
        assert_eq!(
            tags,
            vec![
                Tag {
                    word: "今天", tag: "t"
                },
                Tag {
                    word: "纽约",
                    tag: "ns"
                },
                Tag { word: "的", tag: "uj" },
                Tag {
                    word: "天气", tag: "n"
                },
                Tag {
                    word: "真好", tag: "d"
                },
                Tag { word: "啊", tag: "zg" },
                Tag { word: "，", tag: "x" },
                Tag {
                    word: "京华",
                    tag: "nz"
                },
                Tag {
                    word: "大酒店",
                    tag: "n"
                },
                Tag { word: "的", tag: "uj" },
                Tag {
                    word: "张尧", tag: "x"
                }, // XXX: missing in dict
                Tag {
                    word: "经理", tag: "n"
                },
                Tag { word: "吃", tag: "v" },
                Tag { word: "了", tag: "ul" },
                Tag {
                    word: "一只", tag: "m"
                },
                Tag {
                    word: "北京烤鸭",
                    tag: "n"
                },
                Tag { word: "。", tag: "x" }
            ]
        );
    }

    #[test]
    fn test_tokenize() {
        let jieba = Jieba::new();
        let tokens = jieba.tokenize("南京市长江大桥", TokenizeMode::Default, false);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "南京市",
                    start: 0,
                    end: 3
                },
                Token {
                    word: "长江大桥",
                    start: 3,
                    end: 7
                }
            ]
        );

        let tokens = jieba.tokenize("南京市长江大桥", TokenizeMode::Search, false);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "南京",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "京市",
                    start: 1,
                    end: 3
                },
                Token {
                    word: "南京市",
                    start: 0,
                    end: 3
                },
                Token {
                    word: "长江",
                    start: 3,
                    end: 5
                },
                Token {
                    word: "大桥",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "长江大桥",
                    start: 3,
                    end: 7
                }
            ]
        );

        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, false);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中",
                    start: 2,
                    end: 3
                },
                Token {
                    word: "出",
                    start: 3,
                    end: 4
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, true);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中出",
                    start: 2,
                    end: 4
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );

        let tokens = jieba.tokenize("永和服装饰品有限公司", TokenizeMode::Default, true);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "永和",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "服装",
                    start: 2,
                    end: 4
                },
                Token {
                    word: "饰品",
                    start: 4,
                    end: 6
                },
                Token {
                    word: "有限公司",
                    start: 6,
                    end: 10
                }
            ]
        );
    }

    #[test]
    fn test_userdict() {
        let mut jieba = Jieba::new();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, false);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中",
                    start: 2,
                    end: 3
                },
                Token {
                    word: "出",
                    start: 3,
                    end: 4
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );
        let userdict = "中出 10000";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, false);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中出",
                    start: 2,
                    end: 4
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );
    }

    #[test]
    fn test_userdict_hmm() {
        let mut jieba = Jieba::new();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, true);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中出",
                    start: 2,
                    end: 4
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );
        let userdict = "出了 10000";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, true);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中",
                    start: 2,
                    end: 3
                },
                Token {
                    word: "出了",
                    start: 3,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );
    }

    #[test]
    fn test_userdict_error() {
        let mut jieba = Jieba::empty();
        let userdict = "出了 not_a_int";
        let ret = jieba.load_dict(&mut BufReader::new(userdict.as_bytes()));
        assert!(ret.is_err());
    }

    #[test]
    fn test_load_from_cache_with_user_dict() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("user_dict.cache");

        let user_dict = "测试新词 10000 n\n自定义词 5000 v";
        let jieba = Jieba::load_from_cache_with_user_dict(&cache_path, user_dict).unwrap();

        assert!(jieba.has_word("测试新词"));
        assert!(jieba.has_word("自定义词"));
        assert!(jieba.has_word("中国"));

        // Second load hits mmap fast path (cache file already exists)
        assert!(cache_path.exists());
        let jieba2 = Jieba::load_from_cache_with_user_dict(&cache_path, user_dict).unwrap();
        assert!(jieba2.has_word("测试新词"));
        assert!(jieba2.has_word("中国"));
    }

    #[test]
    fn test_load_from_cache_with_user_dict_override() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("override.cache");

        // "中出" has freq 3 in default dict; override with very high freq
        let user_dict = "中出 100000";
        let jieba = Jieba::load_from_cache_with_user_dict(&cache_path, user_dict).unwrap();

        let words = jieba.cut("我们中出了一个叛徒", false);
        assert_eq!(words, vec!["我们", "中出", "了", "一个", "叛徒"]);
    }

    #[test]
    fn test_load_from_cache_with_user_dict_tag() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("tag.cache");

        let user_dict = "测试新词 10000 nz";
        let jieba = Jieba::load_from_cache_with_user_dict(&cache_path, user_dict).unwrap();

        let tags = jieba.tag("测试新词", false);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].word, "测试新词");
        assert_eq!(tags[0].tag, "nz");
    }

    #[test]
    fn test_load_from_cache_with_user_dict_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("empty_user.cache");

        let jieba = Jieba::load_from_cache_with_user_dict(&cache_path, "").unwrap();

        // Should behave like default dict only
        assert!(jieba.has_word("中国"));
        assert!(jieba.has_word("开源"));
        let words = jieba.cut("我来到北京清华大学", false);
        assert_eq!(words, vec!["我", "来到", "北京", "清华大学"]);
    }

    #[test]
    fn test_load_from_cache_with_user_dict_cut() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("cut.cache");

        let user_dict = "京华大酒店 10000 n";
        let jieba = Jieba::load_from_cache_with_user_dict(&cache_path, user_dict).unwrap();

        let words = jieba.cut("京华大酒店的张尧经理", false);
        assert_eq!(words[0], "京华大酒店");
    }

    #[test]
    fn test_load_from_cache_refactored() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("default.cache");

        let jieba = Jieba::load_from_cache(&cache_path).unwrap();
        assert!(jieba.has_word("中国"));
        assert!(jieba.has_word("开源"));

        // Second load from existing cache
        let jieba2 = Jieba::load_from_cache(&cache_path).unwrap();
        assert!(jieba2.has_word("中国"));
    }

    #[test]
    fn test_suggest_freq() {
        // NOTE: Following behaviors are aligned with original Jieba

        let mut jieba = Jieba::new();
        // These values were calculated by original Jieba
        assert_eq!(jieba.suggest_freq("中出"), 348);
        assert_eq!(jieba.suggest_freq("出了"), 1263);

        // Freq in dict.txt was 3, which became 300 after loading user dict
        let userdict = "中出 300";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        // But it's less than calculated freq 348
        assert_eq!(jieba.suggest_freq("中出"), 348);

        let userdict = "中出 500";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        // Now it's significant enough
        assert_eq!(jieba.suggest_freq("中出"), 500)
    }

    #[test]
    fn test_custom_lower_freq() {
        use std::io::BufReader;

        let mut jieba = Jieba::new();
        let userdict = "测试 10";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        let words = jieba.cut("测试", false);
        assert_eq!(words, vec!["测试"]);
    }

    #[test]
    fn test_cut_dag_no_hmm_against_string_with_sip() {
        use std::io::BufReader;

        let mut jieba = Jieba::empty();
        let userdict = "䶴䶵𦡦 1000\n讥䶯䶰䶱䶲䶳 1000";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();

        let words = jieba.cut("讥䶯䶰䶱䶲䶳䶴䶵𦡦", false);
        assert_eq!(words, vec!["讥䶯䶰䶱䶲䶳", "䶴䶵𦡦"]);
    }

    #[test]
    fn test_load_custom_word_with_underscore() {
        use std::io::BufReader;

        let mut jieba = Jieba::empty();
        let userdict = "田-女士 42 n";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        let words = jieba.cut("市民田-女士急匆匆", false);
        assert_eq!(words, vec!["市", "民", "田-女士", "急", "匆", "匆"]);
    }
}
