// SPDX-License-Identifier: GPL-2.0-only
//! Helpers of the tests that run in a child process: a directory that is
//! removed when the guard drops, also when the test panics, and a child that is
//! killed when it takes too long.

/// The directory of one test, under `/tmp`, named after the process and the
/// clock so that two runs do not meet.
pub struct Scratch(String);

impl Scratch {
    /// Create `/tmp/<prefix>-<pid>-<nanoseconds>`.
    pub fn new(prefix: &str) -> Self {
        let nanoseconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0);
        let path = format!("/tmp/{prefix}-{}-{nanoseconds}", std::process::id());
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// The path of the directory.
    pub fn path(&self) -> &str {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// What a child process came to.
pub struct ChildOutput {
    /// Whether it exited with status zero.
    pub success: bool,
    /// Whether it was killed for taking longer than the deadline.
    pub timed_out: bool,
    /// What it wrote to its standard output.
    pub stdout: String,
    /// What it wrote to its standard error.
    pub stderr: String,
}

/// Run `command` and wait for it at most `seconds`, killing it when the time is
/// up. The output is read while the child runs, so a child that writes more than
/// a pipe holds does not wait for a reader that is waiting for it.
pub fn run_child(mut command: std::process::Command, seconds: u64) -> ChildOutput {
    use std::io::Read;
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut readers = Vec::new();
    for stream in [
        child
            .stdout
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn Read + Send>),
        child
            .stderr
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn Read + Send>),
    ] {
        readers.push(std::thread::spawn(move || {
            let mut text = Vec::new();
            if let Some(mut stream) = stream {
                let _ = stream.read_to_end(&mut text);
            }
            String::from_utf8_lossy(&text).into_owned()
        }));
    }

    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            timed_out = true;
            let _ = child.kill();
            break child.wait().unwrap();
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let stderr = readers.pop().unwrap().join().unwrap();
    let stdout = readers.pop().unwrap().join().unwrap();
    ChildOutput {
        success: status.success() && !timed_out,
        timed_out,
        stdout,
        stderr,
    }
}
