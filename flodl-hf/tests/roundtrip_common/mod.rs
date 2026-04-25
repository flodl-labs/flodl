//! Shared helpers for the `*_roundtrip.rs` integration tests.
//!
//! Each per-family roundtrip test:
//! 1. Loads the HF checkpoint into a flodl model via `from_pretrained`.
//! 2. Saves the model with [`save_safetensors_file_from_graph`].
//! 3. Reads the HF reference safetensors directly from the `hf-hub`
//!    cache.
//! 4. Asserts every flodl-saved key exists on the HF side (after
//!    [`bert_legacy_key_rename`] canonicalisation), shapes agree, and
//!    bytes are bit-exact.
//!
//! Helpers below cover the parts that don't change between families.
//! Each test file's body wires the family-specific `from_pretrained`
//! call and the repo id, then defers to [`run_roundtrip`].

#![allow(dead_code)] // helpers are wired in via `mod roundtrip_common;`

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use hf_hub::api::sync::ApiBuilder;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};

use flodl::Graph;
use flodl_hf::safetensors_io::{
    bert_legacy_key_rename, save_safetensors_file_from_graph, tensor_view_to_f32_vec,
};

/// Resolve the on-disk safetensors path that `from_pretrained` would
/// use for `repo_id`. Mirrors `flodl_hf::hub::fetch_safetensors`:
///
/// 1. `<HF_HOME>/flodl-converted/<repo_id>/model.safetensors` —
///    produced by `fdl flodl-hf convert <repo_id>` for `.bin`-only
///    repos like `microsoft/deberta-v3-base`. When present, this is
///    what flodl actually loaded, so the roundtrip's reference IS
///    this converted file.
/// 2. `api.model(repo_id).get("model.safetensors")` — the normal Hub
///    fetch via `hf-hub`'s on-disk cache.
fn hf_safetensors_path(repo_id: &str) -> PathBuf {
    let hf_home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|h| h.join(".cache").join("huggingface"))
                .unwrap_or_else(|| PathBuf::from("/tmp/huggingface"))
        });
    let converted = hf_home
        .join("flodl-converted")
        .join(repo_id)
        .join("model.safetensors");
    if converted.exists() {
        return converted;
    }

    let api = ApiBuilder::from_env()
        .build()
        .expect("hf-hub init (set HF_HOME for cache location)");
    api.model(repo_id.to_string())
        .get("model.safetensors")
        .unwrap_or_else(|e| {
            panic!(
                "fetch {repo_id}/model.safetensors: {e}\n\
                 If this repo ships only `pytorch_model.bin`, run `fdl flodl-hf convert {repo_id}` first.",
            )
        })
}

/// Decode a safetensors `TensorView` into a flat `Vec<f32>` using the
/// exact same dtype rules flodl applies on load (f32 zero-cost; f16 /
/// bf16 / f64 widened on the host). DeBERTa-v3-base ships its
/// `intermediate.dense.bias` and a few other tensors as F16, so the
/// roundtrip's reference comparison must widen with the same kernel.
fn decode_f32(view: &TensorView<'_>, key: &str) -> Vec<f32> {
    tensor_view_to_f32_vec(view).unwrap_or_else(|e| {
        panic!("decode {key:?} as f32: {e}");
    })
}

/// Run the full roundtrip check for a loaded `graph` against the
/// `repo_id`'s on-disk HF reference. Asserts:
///
/// - every flodl-saved key is present on the HF side after applying
///   [`bert_legacy_key_rename`] to the HF keys;
/// - shapes agree on every shared key;
/// - tensor bytes are bit-exact (f32 in / f32 out, no numerical work
///   between load and save).
///
/// Cleans up the temp save file before asserting so a failed run doesn't
/// leak it.
pub fn run_roundtrip(graph: &Graph, repo_id: &str, family_label: &str) {
    let saved_path = std::env::temp_dir().join(format!(
        "flodl_{family_label}_roundtrip_{}.safetensors",
        std::process::id(),
    ));
    save_safetensors_file_from_graph(graph, &saved_path)
        .unwrap_or_else(|e| panic!("save_safetensors_file_from_graph: {e}"));

    let flodl_bytes = std::fs::read(&saved_path)
        .unwrap_or_else(|e| panic!("read flodl-saved {}: {e}", saved_path.display()));
    let _ = std::fs::remove_file(&saved_path);
    let flodl_st = SafeTensors::deserialize(&flodl_bytes).expect("parse flodl-saved file");

    let hf_path = hf_safetensors_path(repo_id);
    let hf_bytes = std::fs::read(&hf_path)
        .unwrap_or_else(|e| panic!("read HF cache {}: {e}", hf_path.display()));
    let hf_st = SafeTensors::deserialize(&hf_bytes).expect("parse HF reference file");

    // Canonicalise HF keys via the same rename flodl applies on load,
    // so flodl-saved keys (already canonical) line up.
    let mut hf_canonical: HashMap<String, String> = HashMap::new();
    for name in hf_st.names() {
        let canonical = bert_legacy_key_rename(name);
        if let Some(prev) = hf_canonical.insert(canonical.clone(), name.to_string()) {
            panic!(
                "{family_label}: HF-side rename collision: {prev:?} and {name:?} \
                 both map to canonical {canonical:?}",
            );
        }
    }

    let flodl_keys: HashSet<String> = flodl_st.names().iter().map(|s| s.to_string()).collect();
    let hf_canonical_keys: HashSet<&str> = hf_canonical.keys().map(|s| s.as_str()).collect();

    let missing: Vec<&str> = flodl_keys
        .iter()
        .filter(|k| !hf_canonical_keys.contains(k.as_str()))
        .map(|s| s.as_str())
        .collect();
    assert!(
        missing.is_empty(),
        "{family_label}: {} flodl-saved key(s) absent from HF reference: {:?}",
        missing.len(),
        &missing[..missing.len().min(10)],
    );

    let mut diffs: Vec<(String, usize)> = Vec::new();
    let mut shape_mismatches: Vec<(String, Vec<usize>, Vec<usize>)> = Vec::new();
    for fk in &flodl_keys {
        let hf_orig = hf_canonical.get(fk).unwrap();
        let f_view = flodl_st.tensor(fk).unwrap();
        let h_view = hf_st.tensor(hf_orig).unwrap();
        assert_eq!(
            f_view.dtype(),
            Dtype::F32,
            "{family_label}: flodl-saved tensor {fk:?} dtype {:?}, expected F32",
            f_view.dtype(),
        );
        let f_shape = f_view.shape().to_vec();
        let h_shape = h_view.shape().to_vec();
        if f_shape != h_shape {
            shape_mismatches.push((fk.clone(), f_shape, h_shape));
            continue;
        }
        let f_data = decode_f32(&f_view, fk);
        let h_data = decode_f32(&h_view, hf_orig);
        if f_data != h_data {
            let n_mismatches = f_data
                .iter()
                .zip(h_data.iter())
                .filter(|(a, b)| a != b)
                .count();
            diffs.push((fk.clone(), n_mismatches));
        }
    }
    assert!(
        shape_mismatches.is_empty(),
        "{family_label}: shape mismatches: {:?}",
        &shape_mismatches[..shape_mismatches.len().min(10)],
    );
    assert!(
        diffs.is_empty(),
        "{family_label}: {} key(s) differ bit-exact: {:?}",
        diffs.len(),
        &diffs[..diffs.len().min(10)],
    );

    eprintln!(
        "{family_label} roundtrip: {} keys verified bit-exact vs HF reference \
         ({} HF keys total, {} unused on flodl side)",
        flodl_keys.len(),
        hf_canonical_keys.len(),
        hf_canonical_keys.len() - flodl_keys.len(),
    );
}
