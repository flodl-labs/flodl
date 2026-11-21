//! What this `fdl` binary was built from, as `build.rs` embedded it.
//!
//! The crate version moves only at release, so it cannot tell an
//! unreleased build from the release before it. The commit can, and it
//! is also what a remote box can install to run the same fdl.

/// The build's source identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildInfo {
    /// The crate version (`CARGO_PKG_VERSION`).
    pub version: &'static str,
    /// The full commit, when built from flodl's own checkout.
    pub commit: Option<&'static str>,
    /// The fdl sources differed from `commit` at build time.
    pub dirty: bool,
    /// The tag sitting exactly on `commit`, if any.
    pub tag: Option<&'static str>,
}

/// This binary's identity.
pub fn current() -> BuildInfo {
    let nonempty = |v: Option<&'static str>| v.filter(|s| !s.is_empty());
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        commit: nonempty(option_env!("FDL_GIT_COMMIT")),
        dirty: option_env!("FDL_GIT_DIRTY") == Some("1"),
        tag: nonempty(option_env!("FDL_GIT_TAG")),
    }
}

impl BuildInfo {
    /// `flodl-cli 0.8.0`, `flodl-cli 0.8.0 (bfde431)` or
    /// `flodl-cli 0.8.0 (bfde431, dirty)`.
    pub fn version_line(&self) -> String {
        match self.commit {
            None => format!("flodl-cli {}", self.version),
            Some(c) => {
                let short = &c[..c.len().min(7)];
                let dirty = if self.dirty { ", dirty" } else { "" };
                format!("flodl-cli {} ({short}{dirty})", self.version)
            }
        }
    }

    /// True when the published release of this version IS this build:
    /// a build with no checkout behind it (crates.io, a tarball), or a
    /// clean build sitting exactly on the version's tag.
    pub fn is_release(&self) -> bool {
        match self.commit {
            None => true,
            Some(_) => {
                !self.dirty
                    && self.tag.is_some_and(|t| {
                        t == self.version || t.strip_prefix('v') == Some(self.version)
                    })
            }
        }
    }
}

/// The crates whose sources are this binary. Mirrors `FDL_SOURCES` in
/// `build.rs`, minus the workspace manifests.
const SOURCE_DIRS: &[&str] = &["flodl-cli", "flodl-cli-macros", "flodl-hw"];

/// Set once a stale-source check ran in this process tree, by the
/// bootstrap script or by fdl itself, so a nested `fdl` stays quiet.
pub const ENV_STALE_CHECKED: &str = "FDL_STALE_CHECKED";

/// The flodl checkout `start` sits in, if any: the nearest ancestor
/// holding fdl's own crates.
pub fn enclosing_checkout(start: &std::path::Path) -> Option<std::path::PathBuf> {
    start
        .ancestors()
        .find(|dir| {
            SOURCE_DIRS
                .iter()
                .all(|c| dir.join(c).join("Cargo.toml").is_file())
        })
        .map(std::path::Path::to_path_buf)
}

/// A source file of fdl under `checkout` modified after `built`, if one
/// exists. Build output and dot-directories are not sources.
pub fn source_newer_than(
    checkout: &std::path::Path,
    built: std::time::SystemTime,
) -> Option<std::path::PathBuf> {
    fn walk(dir: &std::path::Path, built: std::time::SystemTime) -> Option<std::path::PathBuf> {
        for entry in std::fs::read_dir(dir).ok()?.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "target" || name.starts_with('.') {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                if let Some(hit) = walk(&path, built) {
                    return Some(hit);
                }
            } else if meta.modified().is_ok_and(|m| m > built) {
                return Some(path);
            }
        }
        None
    }
    SOURCE_DIRS
        .iter()
        .find_map(|c| walk(&checkout.join(c), built))
}

/// The notice for a binary older than the fdl sources of the checkout it
/// is run from, or `None` when there is nothing to say.
///
/// An installed `fdl` is a snapshot: nothing rebuilds it when a pull or an
/// edit moves its sources, and the symptom is a fix that visibly merged
/// and visibly does nothing. Only a flodl checkout can be ahead of the
/// binary, so every other directory costs one ancestor walk and no more.
pub fn stale_notice(cwd: &std::path::Path, exe: &std::path::Path) -> Option<String> {
    let checkout = enclosing_checkout(cwd)?;
    let built = std::fs::metadata(exe).ok()?.modified().ok()?;
    source_newer_than(&checkout, built)?;
    // The checkout's own binary is rebuilt through the bootstrap script;
    // any other one was installed, and reinstalling is what updates it.
    let how = if exe.starts_with(&checkout) {
        "FDL_REBUILD=1 ./fdl version"
    } else {
        "cargo install --path flodl-cli"
    };
    Some(format!(
        "fdl: its sources changed since this binary was built; rebuild with: {how}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(commit: Option<&'static str>, dirty: bool, tag: Option<&'static str>) -> BuildInfo {
        BuildInfo {
            version: "0.9.0",
            commit,
            dirty,
            tag,
        }
    }

    const SHA: &str = "bfde4316c0ffee00000000000000000000000000";

    #[test]
    fn version_line_carries_the_short_commit_and_the_dirt() {
        assert_eq!(info(None, false, None).version_line(), "flodl-cli 0.9.0");
        assert_eq!(
            info(Some(SHA), false, None).version_line(),
            "flodl-cli 0.9.0 (bfde431)"
        );
        assert_eq!(
            info(Some(SHA), true, None).version_line(),
            "flodl-cli 0.9.0 (bfde431, dirty)"
        );
    }

    #[test]
    fn a_release_is_a_clean_build_on_its_own_version_tag_or_no_checkout() {
        assert!(info(None, false, None).is_release());
        assert!(info(Some(SHA), false, Some("0.9.0")).is_release());
        assert!(info(Some(SHA), false, Some("v0.9.0")).is_release());
        // On the tag but dirty: the release is not this binary.
        assert!(!info(Some(SHA), true, Some("0.9.0")).is_release());
        // A branch build, and a build on some other tag.
        assert!(!info(Some(SHA), false, None).is_release());
        assert!(!info(Some(SHA), false, Some("0.8.0")).is_release());
    }

    #[test]
    fn this_build_reports_a_consistent_identity() {
        // Whatever build.rs saw, the pieces must agree: no dirt or tag
        // without a commit, and a commit is a full hex sha.
        let me = current();
        assert_eq!(me.version, env!("CARGO_PKG_VERSION"));
        match me.commit {
            None => assert!(!me.dirty && me.tag.is_none()),
            Some(c) => {
                assert!(c.len() == 40 || c.len() == 64, "{c}");
                assert!(c.chars().all(|ch| ch.is_ascii_hexdigit()), "{c}");
            }
        }
    }

    /// A scratch tree shaped like a flodl checkout, removed on drop.
    struct Checkout(std::path::PathBuf);
    impl Checkout {
        fn new(tag: &str) -> Self {
            let root = std::env::temp_dir().join(format!("fdl-stale-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            for c in SOURCE_DIRS {
                std::fs::create_dir_all(root.join(c).join("src")).unwrap();
                std::fs::write(root.join(c).join("Cargo.toml"), "").unwrap();
            }
            Checkout(root)
        }
    }
    impl Drop for Checkout {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Backdate a file by two hours.
    fn age(path: &std::path::Path) {
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(7200);
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(old)
            .unwrap();
    }

    fn hour(offset: i64) -> std::time::SystemTime {
        let d = std::time::Duration::from_secs(3600);
        let now = std::time::SystemTime::now();
        if offset < 0 { now - d } else { now + d }
    }

    #[test]
    fn a_checkout_is_found_from_any_directory_inside_it_and_nowhere_else() {
        let c = Checkout::new("find");
        let deep = c.0.join("flodl-cli").join("src");
        assert_eq!(enclosing_checkout(&deep).as_deref(), Some(c.0.as_path()));
        assert_eq!(enclosing_checkout(&c.0).as_deref(), Some(c.0.as_path()));
        // Outside it, and in a tree missing one of the three crates,
        // there is no checkout.
        assert!(enclosing_checkout(c.0.parent().unwrap()).is_none());
        std::fs::remove_file(c.0.join("flodl-hw").join("Cargo.toml")).unwrap();
        assert!(enclosing_checkout(&deep).is_none());
    }

    #[test]
    fn only_a_source_newer_than_the_binary_counts_and_build_output_never_does() {
        let c = Checkout::new("newer");
        // Everything in the tree was just written: older than a binary
        // built an hour from now, newer than one built an hour ago.
        assert!(source_newer_than(&c.0, hour(1)).is_none());
        assert!(source_newer_than(&c.0, hour(-1)).is_some());

        // Build output and dotfiles are not sources: with the manifests
        // aged, a tree whose only fresh files are those reports nothing.
        let quiet = Checkout::new("output");
        for c in SOURCE_DIRS {
            age(&quiet.0.join(c).join("Cargo.toml"));
        }
        let target = quiet.0.join("flodl-hw").join("target");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("fresh.o"), "").unwrap();
        std::fs::write(quiet.0.join("flodl-hw").join(".cache"), "").unwrap();
        let built = std::time::SystemTime::now() - std::time::Duration::from_secs(60);
        assert_eq!(source_newer_than(&quiet.0, built), None);
    }

    #[test]
    fn the_notice_names_the_rebuild_that_fits_where_the_binary_lives() {
        let c = Checkout::new("notice");
        let cwd = c.0.join("flodl-hw");
        // The checkout's own binary: rebuilt through the bootstrap script.
        let own = c.0.join("target-bin");
        std::fs::write(&own, "").unwrap();
        age(&own);
        let notice = stale_notice(&cwd, &own).expect("sources are newer");
        assert!(notice.contains("FDL_REBUILD=1 ./fdl version"), "{notice}");

        // An installed one, living elsewhere: reinstalled.
        let elsewhere = std::env::temp_dir().join(format!("fdl-stale-bin-{}", std::process::id()));
        std::fs::write(&elsewhere, "").unwrap();
        age(&elsewhere);
        let notice = stale_notice(&cwd, &elsewhere).expect("sources are newer");
        assert!(
            notice.contains("cargo install --path flodl-cli"),
            "{notice}"
        );

        // A binary newer than every source, and a directory outside any
        // checkout, have nothing to report.
        std::fs::File::options()
            .write(true)
            .open(&elsewhere)
            .unwrap()
            .set_modified(hour(1))
            .unwrap();
        assert!(stale_notice(&cwd, &elsewhere).is_none());
        assert!(stale_notice(&std::env::temp_dir(), &own).is_none());
        let _ = std::fs::remove_file(&elsewhere);
    }
}
