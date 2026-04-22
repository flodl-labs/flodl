//! flodl-hf playground.
//!
//! `AutoModel` dispatches on `config.json`'s `model_type` so the same
//! code path serves BERT, RoBERTa, and DistilBERT. Pass a HuggingFace
//! repo id as the first argument; with no argument, defaults to
//! `cardiffnlp/twitter-roberta-base-sentiment-latest`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release
//! cargo run --release -- bert-base-uncased
//! cargo run --release -- lxyuan/distilbert-base-multilingual-cased-sentiments-student
//! ```
//!
//! See README.md for feature flavors (offline / vision-only), the
//! `fdl flodl-hf convert` workflow for `.bin`-only repos, and how to
//! point at your own model.

use flodl_hf::models::auto::AutoModelForSequenceClassification;

fn main() -> flodl::Result<()> {
    let repo_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cardiffnlp/twitter-roberta-base-sentiment-latest".to_string());

    eprintln!("loading {repo_id} via AutoModel...");
    let clf = AutoModelForSequenceClassification::from_pretrained(&repo_id)?;

    let texts = &[
        "I love this framework so much",
        "I can't believe they shut down the old service",
        "I'm a little anxious about the release",
    ];
    let preds = clf.predict(texts)?;

    for (text, row) in texts.iter().zip(&preds) {
        let (top_label, top_score) = row
            .first()
            .expect("predict returns at least one label");
        println!("{text:?}");
        println!("  top: {top_label} ({top_score:.3})");
        for (label, score) in row.iter().skip(1).take(2) {
            println!("  ..  {label} ({score:.3})");
        }
    }
    Ok(())
}
