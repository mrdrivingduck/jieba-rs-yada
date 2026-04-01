use std::env;
use std::process;

use jieba_rs_yada::Jieba;

/// Default cache file path for mmap-based dictionary sharing across processes.
const CACHE_PATH: &str = "/tmp/jieba-rs-yada.dict.cache";

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments: <num_children> [--mmap]
    let mut num_children: Option<usize> = None;
    let mut use_mmap = false;

    for arg in &args[1..] {
        if arg == "--mmap" {
            use_mmap = true;
        } else if num_children.is_none() {
            num_children = arg.parse::<usize>().ok().filter(|&n| n > 0);
        }
    }

    let num_children = match num_children {
        Some(n) => n,
        None => {
            eprintln!("Usage: {} <num_children> [--mmap]", args[0]);
            eprintln!();
            eprintln!("  <num_children>  Number of child processes to fork (positive integer)");
            eprintln!("  --mmap          Use mmap-based cache loading (children generate cache concurrently)");
            eprintln!("                  Without --mmap, each child calls Jieba::new() independently");
            process::exit(1);
        }
    };

    let mode_desc = if use_mmap { "mmap (load_from_cache)" } else { "heap (Jieba::new)" };
    println!(
        "[parent pid={}] Mode: {mode_desc}. Forking {num_children} child processes...",
        unsafe { libc::getpid() }
    );

    // When using mmap mode, delete any stale cache file so that children will
    // race to generate it — this is the scenario we want to validate.
    if use_mmap {
        let _ = std::fs::remove_file(CACHE_PATH);
        println!("[parent] Removed stale cache (if any). Children will generate it concurrently.");
    }

    let mut child_pids: Vec<libc::pid_t> = Vec::with_capacity(num_children);

    for child_index in 0..num_children {
        let pid = unsafe { libc::fork() };
        match pid {
            -1 => {
                eprintln!("[parent] fork() failed for child {child_index}");
                for &cpid in &child_pids {
                    unsafe { libc::kill(cpid, libc::SIGTERM); }
                }
                process::exit(1);
            }
            0 => {
                // ---- Child process ----
                run_child(child_index, use_mmap);
            }
            child_pid => {
                // ---- Parent process ----
                child_pids.push(child_pid);
            }
        }
    }

    println!("[parent] All {num_children} children forked. Press Ctrl+C to stop.");

    install_signal_handler();

    // Wait for all children to exit.
    let mut exited_count = 0;
    while exited_count < child_pids.len() {
        let mut status: libc::c_int = 0;
        let waited_pid = unsafe { libc::waitpid(-1, &mut status, 0) };
        if waited_pid > 0 {
            exited_count += 1;
            println!("[parent] Child pid={waited_pid} exited ({exited_count}/{})", child_pids.len());
        } else if waited_pid == -1 {
            // ECHILD means no more children
            break;
        }
    }

    println!("[parent] All children exited. Parent exiting.");
}

/// Runs in the child process: load dictionary, do one segmentation, report, then sleep forever.
///
/// - `use_mmap = false`: calls `Jieba::new()` (each child builds its own heap copy)
/// - `use_mmap = true`: calls `Jieba::load_from_cache()` (children race to generate the
///   cache file, then all mmap the same file to share physical memory pages)
fn run_child(child_index: usize, use_mmap: bool) -> ! {
    let my_pid = unsafe { libc::getpid() };

    let jieba = if use_mmap {
        println!("[child {child_index} pid={my_pid}] Loading jieba via load_from_cache({CACHE_PATH})...");
        Jieba::load_from_cache(CACHE_PATH).expect("failed to load dictionary from cache")
    } else {
        println!("[child {child_index} pid={my_pid}] Loading jieba via Jieba::new()...");
        Jieba::new()
    };

    // Perform one segmentation
    let test_sentence = "我们中出了一个叛徒，今天纽约的天气真好啊";
    let words = jieba.cut(test_sentence, true);
    println!(
        "[child {child_index} pid={my_pid}] Segmentation done: {:?}",
        words
    );

    println!("[child {child_index} pid={my_pid}] Ready. Sleeping until signaled...");

    // Sleep forever (not burning CPU) — pause() blocks until a signal is delivered.
    loop {
        unsafe { libc::pause(); }
    }
}

/// Install a minimal signal handler for SIGINT and SIGTERM.
/// Since all forked children share the same process group, Ctrl+C sends
/// SIGINT to the entire group — children receive it directly from the terminal.
fn install_signal_handler() {
    unsafe {
        libc::signal(libc::SIGINT, signal_handler as usize);
        libc::signal(libc::SIGTERM, signal_handler as usize);
    }
}

/// Minimal signal handler — the signal delivery interrupts waitpid()/pause().
extern "C" fn signal_handler(_sig: libc::c_int) {
    // Intentionally empty.
}
