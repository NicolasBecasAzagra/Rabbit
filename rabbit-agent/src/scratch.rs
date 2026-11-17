// SPDX-License-Identifier: GPL-2.0-only
//! A directory for a test, removed when the guard drops, also when the test
//! panics.

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
