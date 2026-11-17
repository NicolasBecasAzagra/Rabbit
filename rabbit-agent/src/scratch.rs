// SPDX-License-Identifier: GPL-2.0-only
//! Helpers of the tests that run in a child process: a directory that is
//! removed when the guard drops, also when the test panics, and a child that is
//! killed when it takes too long.

use std::io;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};

/// The private parent of the directories of the tests, under `/tmp`. The name
/// is short because a path of the manifest holds at most sixty four bytes.
const PARENT: &str = "/tmp/rb-tests";

/// The directory of one test, created exclusively under a private parent: a
/// name that already exists, as a directory or as a link, is refused, so a
/// program that guessed the name cannot have prepared it.
pub struct Scratch(String);

impl Scratch {
    /// Create `/tmp/rb-tests/<prefix>-<pid>[-<n>]` with mode 0700.
    pub fn new(prefix: &str) -> Self {
        Self::in_parent(PARENT, prefix).unwrap()
    }

    /// Create the directory under `parent`, which is made private if it is not
    /// there and refused if it is a link, is not ours or is open to others.
    pub fn in_parent(parent: &str, prefix: &str) -> io::Result<Self> {
        match std::fs::DirBuilder::new().mode(0o700).create(parent) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
        // `symlink_metadata` does not follow a link, so a link in the place of
        // the parent is seen as a link and refused.
        let metadata = std::fs::symlink_metadata(parent)?;
        let ours = std::fs::metadata("/proc/self")?.uid();
        if !metadata.is_dir() || metadata.uid() != ours || metadata.mode() & 0o077 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("{parent} is not a private directory of this user"),
            ));
        }
        for attempt in 0..100 {
            let name = if attempt == 0 {
                format!("{prefix}-{}", std::process::id())
            } else {
                format!("{prefix}-{}-{attempt}", std::process::id())
            };
            let path = format!("{parent}/{name}");
            // Not `create_dir_all`: this fails when the name exists.
            match std::fs::DirBuilder::new().mode(0o700).create(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "every name of the directory is taken",
        ))
    }

    /// Create exactly `parent/name`, and fail when it exists.
    #[cfg(test)]
    fn exclusive(parent: &str, name: &str) -> io::Result<Self> {
        let path = format!("{parent}/{name}");
        std::fs::DirBuilder::new().mode(0o700).create(&path)?;
        Ok(Self(path))
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
/// up. The output is read while the child runs, so a child that writes more
/// than a pipe holds does not wait for a reader that is waiting for it.
///
/// A reader that a grandchild keeps open is not waited for: after the child is
/// gone the readers get two seconds, and what they have read is the output. The
/// thread that is still reading is left to end with its pipe.
pub fn run_child(mut command: std::process::Command, seconds: u64) -> ChildOutput {
    use std::io::Read;
    use std::process::Stdio;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut captured = Vec::new();
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
        let text = Arc::new(Mutex::new(Vec::<u8>::new()));
        let done = Arc::new(AtomicBool::new(false));
        let (shared, finished) = (Arc::clone(&text), Arc::clone(&done));
        std::thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            if let Some(mut stream) = stream {
                while let Ok(length) = stream.read(&mut chunk) {
                    if length == 0 {
                        break;
                    }
                    shared.lock().unwrap().extend_from_slice(&chunk[..length]);
                }
            }
            finished.store(true, Ordering::Relaxed);
        });
        captured.push((text, done));
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

    let grace = Instant::now() + Duration::from_secs(2);
    while Instant::now() < grace
        && !captured
            .iter()
            .all(|(_, done)| done.load(Ordering::Relaxed))
    {
        std::thread::sleep(Duration::from_millis(20));
    }
    let read =
        |index: usize| String::from_utf8_lossy(&captured[index].0.lock().unwrap()).into_owned();
    ChildOutput {
        success: status.success() && !timed_out,
        timed_out,
        stdout: read(0),
        stderr: read(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A parent of its own, so the test does not meet the real one.
    fn private_parent() -> Scratch {
        Scratch::new("scratch-test")
    }

    #[test]
    fn a_name_that_exists_is_refused() {
        let parent = private_parent();
        let taken = format!("{}/taken", parent.path());
        std::fs::create_dir(&taken).unwrap();
        assert!(Scratch::exclusive(parent.path(), "taken").is_err());
        assert!(Scratch::exclusive(parent.path(), "free").is_ok());
    }

    #[test]
    fn a_name_that_is_a_link_is_refused() {
        let parent = private_parent();
        let target = format!("{}/target", parent.path());
        std::fs::create_dir(&target).unwrap();
        let link = format!("{}/link", parent.path());
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(Scratch::exclusive(parent.path(), "link").is_err());
    }

    #[test]
    fn a_parent_that_is_a_link_or_open_to_others_is_refused() {
        let parent = private_parent();
        let real = format!("{}/real", parent.path());
        std::fs::create_dir(&real).unwrap();
        let link = format!("{}/link", parent.path());
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(Scratch::in_parent(&link, "x").is_err());

        let open = format!("{}/open", parent.path());
        std::fs::DirBuilder::new()
            .mode(0o755)
            .create(&open)
            .unwrap();
        // The mode of a new directory is cut by the umask; make it open.
        std::fs::set_permissions(&open, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .unwrap();
        assert!(Scratch::in_parent(&open, "x").is_err());
    }

    #[test]
    fn a_directory_of_a_test_is_private_and_removed_when_it_drops() {
        let path = {
            let scratch = Scratch::new("scratch-test");
            let metadata = std::fs::symlink_metadata(scratch.path()).unwrap();
            assert!(metadata.is_dir());
            assert_eq!(metadata.mode() & 0o077, 0);
            scratch.path().to_string()
        };
        assert!(std::fs::symlink_metadata(path).is_err());
    }

    #[test]
    fn two_directories_of_one_process_do_not_meet() {
        let first = Scratch::new("scratch-test");
        let second = Scratch::new("scratch-test");
        assert_ne!(first.path(), second.path());
    }
}
