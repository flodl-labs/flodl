//! Sequence classification with `RobertaForSequenceClassification`.
//!
//! Closed-loop demo: download a RoBERTa-based sentiment classifier,
//! run it on a few prompts, and print per-label probabilities.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example roberta-classify
//! # or directly:
//! cargo run --release --example roberta_classify
//! ```

use flodl_hf::models::roberta::RobertaForSequenceClassification;

fn main() -> flodl::Result<()> {
    // Three sentiment classes: negative, neutral, positive.
    let clf = RobertaForSequenceClassification::from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
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
