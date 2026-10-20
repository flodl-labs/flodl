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
}
