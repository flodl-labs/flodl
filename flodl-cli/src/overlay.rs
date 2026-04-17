//! Multi-environment configuration overlays.
//!
//! An `fdl.yml` project manifest can be layered with per-environment files
//! (e.g. `fdl.local.yml`, `fdl.ci.yml`, `fdl.cloud.yml`). When an environment
//! is active, its file is deep-merged on top of the base config before the
//! strongly-typed [`ProjectConfig`](crate::config::ProjectConfig) /
//! [`CommandConfig`](crate::config::CommandConfig) deserialization runs.
//!
//! # Merge rules
//!
//! - **Maps**: deep-merge. Recurse into nested maps; overlay keys win.
//! - **Scalars**: replace. Overlay value takes over.
//! - **Lists**: replace entirely. (Order is contentious — append/prepend
//!   modes cause more debugging pain than they save.)
//! - **`null` deletes**: a key set to `null` in the overlay removes it from
//!   the merged map (not "write null"). Useful for "reset to defaults in
//!   this env."
//!
//! # Discovery
//!
//! Sibling files matching `fdl.<env>.{yml,yaml,json}` alongside the base
//! config. The `<env>` token is the first-arg env selector.

use std::path::{Path, PathBuf};

use serde_yaml::{Mapping, Value};

// ── Deep-merge ──────────────────────────────────────────────────────────

/// Deep-merge `over` onto `base`. Maps recurse; scalars and lists replace;
/// `null` values in a map context delete the key from the result.
///
/// Non-Mapping destinations are replaced wholesale when the overlay is a
/// Mapping too — i.e. no cross-type merging, the newer value wins.
pub fn deep_merge(base: Value, over: Value) -> Value {
    match (base, over) {
        (Value::Mapping(base_map), Value::Mapping(over_map)) => {
            Value::Mapping(merge_mapping(base_map, over_map))
        }
        // Scalar, sequence, or type-change: overlay replaces base.
        (_, over) => over,
    }
}

fn merge_mapping(mut base: Mapping, over: Mapping) -> Mapping {
    for (k, v) in over {
        if matches!(v, Value::Null) {
            base.remove(&k);
            continue;
        }
        match base.remove(&k) {
            Some(existing) => {
                base.insert(k, deep_merge(existing, v));
            }
            None => {
                base.insert(k, v);
            }
        }
    }
    base
}

/// Merge a chain of layers left-to-right. The first is the base; each
/// subsequent layer is merged on top of the running result.
pub fn merge_layers<I>(layers: I) -> Value
where
    I: IntoIterator<Item = Value>,
{
    layers
        .into_iter()
        .reduce(deep_merge)
        .unwrap_or(Value::Null)
}

// ── Discovery ───────────────────────────────────────────────────────────

/// Config filename extensions in preference order. Mirrors `config::CONFIG_NAMES`
/// but exposed here so overlay lookup matches sibling base files.
const EXTENSIONS: &[&str] = &["yml", "yaml", "json"];

/// Find a sibling overlay for `env` next to `base_config`.
///
/// `base_config` should be the resolved path to the base `fdl.yml` (not a
/// directory). Returns `Some(path)` if `fdl.<env>.<ext>` exists for any
/// supported extension, `None` otherwise.
pub fn find_env_file(base_config: &Path, env: &str) -> Option<PathBuf> {
    let dir = base_config.parent()?;
    for ext in EXTENSIONS {
        let candidate = dir.join(format!("fdl.{env}.{ext}"));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// List every environment overlay discoverable beside the base config.
///
/// Returns env names (without `fdl.` prefix or extension), sorted. Duplicate
/// names across extensions are de-duplicated — the first-found wins, matching
/// [`find_env_file`] precedence.
pub fn list_envs(base_config: &Path) -> Vec<String> {
    let Some(dir) = base_config.parent() else {
        return Vec::new();
    };
    let entries = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut envs = std::collections::BTreeSet::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let Some(stripped) = name_str.strip_prefix("fdl.") else {
            continue;
        };
        // Must have at least one `.` separating env name from extension.
        let Some((env, ext)) = stripped.rsplit_once('.') else {
            continue;
        };
        if env.is_empty() || !EXTENSIONS.contains(&ext) {
            continue;
        }
        envs.insert(env.to_string());
    }
    envs.into_iter().collect()
}

/// Load a YAML/JSON file as a [`Value`]. Extension-based dispatch matches
/// [`crate::config::parse`].
pub fn load_value(path: &Path) -> Result<Value, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("yaml");
    match ext {
        "json" => serde_json::from_str::<Value>(&content)
            .map_err(|e| format!("{}: {}", path.display(), e)),
        _ => serde_yaml::from_str::<Value>(&content)
            .map_err(|e| format!("{}: {}", path.display(), e)),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml(s: &str) -> Value {
        serde_yaml::from_str(s).expect("test fixture must parse")
    }

    #[test]
    fn scalar_over_scalar_replaces() {
        let base = yaml("42");
        let over = yaml("99");
        assert_eq!(deep_merge(base, over), yaml("99"));
    }

    #[test]
    fn map_keys_deep_merge() {
        let base = yaml(
            r"
            a: 1
            nested:
              x: one
              y: two
            ",
        );
        let over = yaml(
            r"
            nested:
              y: TWO
              z: three
            b: 2
            ",
        );
        let expected = yaml(
            r"
            a: 1
            b: 2
            nested:
              x: one
              y: TWO
              z: three
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn lists_replace_not_append() {
        let base = yaml(
            r"
            items: [a, b, c]
            ",
        );
        let over = yaml(
            r"
            items: [x, y]
            ",
        );
        let expected = yaml(
            r"
            items: [x, y]
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn null_in_overlay_deletes_key() {
        let base = yaml(
            r"
            ddp:
              policy: cadence
              anchor: 3
            training:
              epochs: 10
            ",
        );
        let over = yaml(
            r"
            ddp: ~
            training:
              epochs: 20
            ",
        );
        // `ddp: null` removes the whole block; training.epochs updates.
        let expected = yaml(
            r"
            training:
              epochs: 20
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn null_leaf_removes_single_key() {
        let base = yaml(
            r"
            ddp:
              policy: cadence
              anchor: 3
            ",
        );
        let over = yaml(
            r"
            ddp:
              anchor: ~
            ",
        );
        let expected = yaml(
            r"
            ddp:
              policy: cadence
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn overlay_adds_new_top_level_key() {
        let base = yaml("a: 1");
        let over = yaml("b: 2");
        let expected = yaml(
            r"
            a: 1
            b: 2
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn merge_chain_three_layers() {
        let l1 = yaml("a: 1\nb: 1");
        let l2 = yaml("b: 2\nc: 2");
        let l3 = yaml("c: 3");
        let got = merge_layers(vec![l1, l2, l3]);
        let expected = yaml(
            r"
            a: 1
            b: 2
            c: 3
            ",
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn type_change_overlay_replaces_wholesale() {
        let base = yaml(
            r"
            ddp:
              policy: cadence
            ",
        );
        let over = yaml(
            r"
            ddp: solo-0
            ",
        );
        let expected = yaml(
            r"
            ddp: solo-0
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn type_change_scalar_base_mapping_overlay_replaces() {
        // Symmetry with `type_change_overlay_replaces_wholesale`: when
        // the base is a scalar and the overlay is a mapping, the mapping
        // wins wholesale. No attempt at cross-type merging.
        let base = yaml(
            r"
            ddp: solo-0
            ",
        );
        let over = yaml(
            r"
            ddp:
              policy: cadence
              anchor: 3
            ",
        );
        let expected = yaml(
            r"
            ddp:
              policy: cadence
              anchor: 3
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn list_envs_discovers_sibling_overlays() {
        let tmp = tempdir();
        std::fs::write(tmp.path().join("fdl.yml"), "description: base").unwrap();
        std::fs::write(tmp.path().join("fdl.ci.yml"), "description: ci").unwrap();
        std::fs::write(tmp.path().join("fdl.cloud.yaml"), "description: cloud").unwrap();
        std::fs::write(tmp.path().join("fdl.prod.json"), "{}").unwrap();
        // Decoys — must NOT be listed.
        std::fs::write(tmp.path().join("fdl.yml.example"), "").unwrap();
        std::fs::write(tmp.path().join("other.ci.yml"), "").unwrap();
        std::fs::write(tmp.path().join("fdl.yml.bak"), "").unwrap();

        let envs = list_envs(&tmp.path().join("fdl.yml"));
        assert_eq!(envs, vec!["ci".to_string(), "cloud".into(), "prod".into()]);
    }

    #[test]
    fn find_env_file_respects_extension_precedence() {
        let tmp = tempdir();
        std::fs::write(tmp.path().join("fdl.yml"), "").unwrap();
        std::fs::write(tmp.path().join("fdl.ci.yml"), "# yml wins").unwrap();
        std::fs::write(tmp.path().join("fdl.ci.yaml"), "# yaml loses").unwrap();

        let got = find_env_file(&tmp.path().join("fdl.yml"), "ci").unwrap();
        assert_eq!(got.file_name().unwrap().to_str(), Some("fdl.ci.yml"));
    }

    #[test]
    fn find_env_file_missing_returns_none() {
        let tmp = tempdir();
        std::fs::write(tmp.path().join("fdl.yml"), "").unwrap();
        assert!(find_env_file(&tmp.path().join("fdl.yml"), "nope").is_none());
    }

    // Tiny tempdir helper — standalone so we don't pull in the tempfile crate.
    fn tempdir() -> TempDir {
        TempDir::new()
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let base = std::env::temp_dir();
            let unique = format!(
                "flodl-overlay-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            );
            let dir = base.join(unique);
            std::fs::create_dir_all(&dir).expect("tempdir creation");
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
