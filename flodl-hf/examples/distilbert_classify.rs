//! Sequence classification with `DistilBertForSequenceClassification`.
//!
//! Closed-loop demo: download a DistilBERT-based multilingual
//! sentiment classifier (3 classes — positive / neutral / negative),
//! run it on a few prompts, and print per-label probabilities.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example distilbert-classify
//! # or directly:
//! cargo run --release --example distilbert_classify
//! ```

use flodl_hf::models::distilbert::DistilBertForSequenceClassification;

fn main() -> flodl::Result<()> {
    let clf = DistilBertForSequenceClassification::from_pretrained(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )?;

    let texts = &[
        "I really love this new Rust framework",
        "Support still feels broken after the update",
        "The release notes landed this morning",
    ];
    let preds = clf.predict(texts)?;

    for (text, row) in texts.iter().zip(&preds) {
        let (top_label, top_score) = row.first().expect("predict returns at least one label");
        println!("{text:?}");
        println!("  top: {top_label} ({top_score:.3})");
        for (label, score) in row.iter().skip(1).take(2) {
            println!("  ..  {label} ({score:.3})");
        }
    }
    Ok(())
}
