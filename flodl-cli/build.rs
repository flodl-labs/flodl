//! Embeds the git commit this `fdl` was built from.
//!
//! The crate version cannot identify a build: it moves only at release,
//! so a binary built from an unreleased branch reports the same version
//! as the release before it. A walk-in provisioned from such a build
//! must run that same code, and the commit is what can be installed
//! remotely (`cargo install --git <repo> --rev <commit>`).
//!
//! Three values reach the crate as compile-time env:
//! - `FDL_GIT_COMMIT`: the full commit, or unset.
//! - `FDL_GIT_DIRTY`: `1` when the sources fdl is built from differ from
//!   that commit, so installing the commit elsewhere yields another fdl.
//! - `FDL_GIT_TAG`: the tag sitting exactly on the commit, if any.
//!
//! Nothing is embedded unless the git toplevel is flodl's own checkout
//! with this crate at `flodl-cli/`: a packaged copy under
//! `target/package/` or a registry copy inside an unrelated repository
//! would otherwise report that repository's commit. Any git failure
//! (no git, not a checkout, an ownership refusal) embeds nothing and
//! never fails the build.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The paths fdl is compiled from, relative to the checkout root. A
/// change anywhere else does not change the binary, so it is not dirt.
const FDL_SOURCES: &[&str] = &[
    "flodl-cli",
    "flodl-cli-macros",
    "flodl-hw",
    "Cargo.toml",
    "Cargo.lock",
];

/// Runs git without optional locks: a plain `git status` refreshes and
/// rewrites the index, which would retrigger this script on every build.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let Some(top) = git(&manifest, &["rev-parse", "--show-toplevel"]).map(PathBuf::from) else {
        return;
    };
    let ours = match (
        top.join("flodl-cli").canonicalize(),
        manifest.canonicalize(),
    ) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !ours {
        return;
    }
    let Some(commit) = git(&top, &["rev-parse", "HEAD"]).filter(|c| !c.is_empty()) else {
        return;
    };
    let mut status = vec!["status", "--porcelain", "--"];
    status.extend(FDL_SOURCES);
    // A status that cannot be read counts as dirty: claiming clean would
    // let a remote install a commit that is not this binary.
    let dirty = git(&top, &status).is_none_or(|s| !s.is_empty());
    let tag = git(&top, &["describe", "--exact-match", "--tags", "HEAD"]).unwrap_or_default();

    println!("cargo:rustc-env=FDL_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=FDL_GIT_DIRTY={}", u8::from(dirty));
    println!("cargo:rustc-env=FDL_GIT_TAG={tag}");

    // What moves the three values: HEAD (checkout), the ref it names and
    // packed-refs (commit), the tag refs (tagging), and the sources (dirt).
    let mut watch: Vec<PathBuf> = FDL_SOURCES.iter().map(|p| top.join(p)).collect();
    let mut git_paths = vec![
        "HEAD".to_string(),
        "packed-refs".to_string(),
        "refs/tags".to_string(),
    ];
    if let Some(head_ref) = git(&top, &["symbolic-ref", "-q", "HEAD"]) {
        git_paths.push(head_ref);
    }
    for p in &git_paths {
        if let Some(abs) = git(
            &top,
            &["rev-parse", "--path-format=absolute", "--git-path", p],
        ) {
            watch.push(PathBuf::from(abs));
        }
    }
    // Only paths that exist: cargo treats a missing one as changed, which
    // would rerun this script (and rebuild the crate) on every build.
    for p in watch.iter().filter(|p| p.exists()) {
        println!("cargo:rerun-if-changed={}", p.display());
    }
}
