//! Where flodl keeps what one box manages for itself: the dataset cache
//! and the telemetry spill.
//!
//! `$FLODL_HOME` when set, else `~/.flodl`. The same rule `fdl` follows
//! for the libtorch variants and fetched sources it manages, so one
//! variable relocates everything, and a container can be pointed at a
//! mounted directory without pretending to have a home.

use std::ffi::OsString;
use std::path::PathBuf;

/// The node-local root, or `None` when neither `FLODL_HOME` nor `HOME`
/// names one. Callers pick their own fallback: what losing the directory
/// costs differs between a cache and a crash log.
pub(crate) fn flodl_home() -> Option<PathBuf> {
    resolve(std::env::var_os("FLODL_HOME"), std::env::var_os("HOME"))
}

/// An empty value counts as unset: it would otherwise resolve relative to
/// the working directory, which is never what an empty variable means.
fn resolve(flodl_home: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    if let Some(root) = flodl_home.filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(root));
    }
    home.filter(|v| !v.is_empty())
        .map(|h| PathBuf::from(h).join(".flodl"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(s: &str) -> Option<OsString> {
        Some(OsString::from(s))
    }

    #[test]
    fn flodl_home_wins_over_home() {
        assert_eq!(
            resolve(os("/mnt/flodl"), os("/home/u")),
            Some(PathBuf::from("/mnt/flodl"))
        );
    }

    #[test]
    fn home_gets_the_dot_directory() {
        assert_eq!(
            resolve(None, os("/home/u")),
            Some(PathBuf::from("/home/u/.flodl"))
        );
    }

    #[test]
    fn an_empty_flodl_home_falls_through_to_home() {
        assert_eq!(
            resolve(os(""), os("/home/u")),
            Some(PathBuf::from("/home/u/.flodl"))
        );
    }

    #[test]
    fn nothing_set_is_none() {
        assert_eq!(resolve(None, None), None);
        assert_eq!(resolve(os(""), os("")), None);
    }
}
