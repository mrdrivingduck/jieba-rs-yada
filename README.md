# jieba-rs-yada

`jieba-rs-yada` is a Rust fork of [`jieba-rs`](https://crates.io/crates/jieba-rs).

This fork keeps the original Chinese segmentation and keyword extraction capabilities, with a stronger focus on memory sharing for multi-process deployments.

## Core Changes In This Fork

1. Renamed crate/package to `jieba-rs-yada` (version line starts at `0.1.0`).
2. Replaced the Double Array Trie backend from [`cedarwood`](https://crates.io/crates/cedarwood) to [`yada`](https://crates.io/crates/yada).
3. Added dictionary cache APIs:
   - `save_cache(path)`: atomically writes dictionary + DAT to a cache file.
   - `load_from_cache(path)`: loads cache via zero-copy `mmap`.
4. Improved multi-process safety:
   - Uses a `.lock` sidecar file to prevent duplicate cache builds under concurrency.
   - Uses temp-file + `rename` for atomic replacement (no partial reads).
5. Added `examples/multiprocess` to demonstrate and validate mmap-based memory sharing.

## Use Cases

- Chinese word segmentation (`cut`, `cut_all`, `cut_for_search`)
- POS tagging
- Keyword extraction (`tfidf` / `textrank` features)
- Multi-process services (e.g. prefork workers) that need lower per-process dictionary memory

## Installation

```toml
[dependencies]
jieba-rs-yada = "0.1"
```

Enable keyword extraction:

```toml
[dependencies]
jieba-rs-yada = { version = "0.1", features = ["tfidf", "textrank"] }
```

## Quick Start

```rust
use jieba_rs_yada::Jieba;

fn main() {
    let jieba = Jieba::new();
    let words = jieba.cut("我们中出了一个叛徒", false);
    assert_eq!(words, vec!["我们", "中", "出", "了", "一个", "叛徒"]);
}
```

## Multi-Process Dictionary Sharing (mmap)

### Option 1: Prewarm and Persist Cache

```rust
use jieba_rs_yada::Jieba;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let jieba = Jieba::new();
    jieba.save_cache("/tmp/jieba-rs-yada.dict.cache")?;
    Ok(())
}
```

### Option 2: Load On Demand At Runtime

```rust
use jieba_rs_yada::Jieba;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let jieba = Jieba::load_from_cache("/tmp/jieba-rs-yada.dict.cache")?;
    let words = jieba.cut("测试分词", true);
    println!("{:?}", words);
    Ok(())
}
```

If the cache file does not exist, `load_from_cache` builds it under a file lock and then loads it. Later processes directly `mmap` the same file and share physical pages.

## Validation Utility

- `examples/weicheng`: baseline segmentation performance example
- `examples/multiprocess`: a validation utility to verify mmap-based dictionary sharing works correctly under concurrent multi-process startup

Run the mmap validation utility:

```bash
cargo run -p jieba-rs-yada-multiprocess -- 8 --mmap
```

Press `Ctrl+C` to stop the parent process and all forked children.

Compare without mmap:

```bash
cargo run -p jieba-rs-yada-multiprocess -- 8
```

On Linux, you can inspect process-level `PSS` ([Proportional Set Size](https://en.wikipedia.org/wiki/Proportional_set_size)) with [`smem`](https://www.selenic.com/smem/) to verify whether dictionary pages are actually shared:

```bash
smem -P jieba-rs-yada-multiprocess -c "pid command uss rss pss" -k
```

With `--mmap`, `PSS` per worker should typically be lower than the non-mmap run.

## API Overview

- Segmentation: `cut` / `cut_all` / `cut_for_search`
- Token offsets: `tokenize`
- POS tagging: `tag`
- Custom dictionary loading: `load_dict`
- **Cache APIs: `save_cache` / `load_from_cache`**

## C API

This repository includes `jieba-capi` (`cdylib`) for C ABI integration. See `capi/`.

## Benchmark

```bash
cargo bench --all-features
```

## Acknowledgements

This project builds on `jieba-rs`. Thanks to the upstream maintainers and community contributors.

## License

MIT. See [LICENSE](./LICENSE).
