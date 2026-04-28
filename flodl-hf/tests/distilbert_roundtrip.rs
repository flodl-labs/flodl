//! Roundtrip: HF → flodl load → flodl save → bytes-equality vs HF.
//!
//! Inverse of `distilbert_parity.rs`. Same `_live` mechanics — pulls
//! real `distilbert/distilbert-base-uncased` weights from the Hub
//! (cached after the first run).
//!
//! See `roundtrip_common::run_roundtrip` for the assertion shape.

mod roundtrip_common;

use flodl_hf::models::distilbert::DistilBertModel;

#[test]
#[ignore = "live: roundtrip via safetensors save; requires network + hf-hub cache"]
fn distilbert_roundtrip_vs_pytorch_live() {
    let graph = DistilBertModel::from_pretrained("distilbert/distilbert-base-uncased").unwrap();
    roundtrip_common::run_roundtrip(&graph, "distilbert/distilbert-base-uncased", "distilbert");
}
