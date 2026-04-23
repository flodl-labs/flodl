//! Fine-tune a DistilBERT sentiment classifier on a tiny inline
//! dataset. Full fine-tuning (backbone + classifier), CPU-only,
//! ~30 seconds after the one-time weight download (~250 MB cached
//! via `hf-hub`).
//!
//! Starting point: `distilbert-base-uncased-finetuned-sst-2-english`,
//! a two-class positive / negative sentiment head. The example
//! continues training on a hand-crafted domain-specific dataset -
//! the canonical "calibrate a generic sentiment model to your own
//! corpus" pattern. Same training loop scales to a real dataset by
//! swapping the inline pairs for a proper [`DataSet`](flodl::DataSet)
//! implementation.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example distilbert-finetune
//! # or directly:
//! cargo run --release --example distilbert_finetune
//! ```

use flodl::{clip_grad_norm, Adam, Device, Module, Optimizer, Result, Tensor, Variable};
use flodl_hf::models::distilbert::DistilBertForSequenceClassification;
use flodl_hf::tokenizer::HfTokenizer;

fn main() -> Result<()> {
    let model_repo = "distilbert-base-uncased-finetuned-sst-2-english";
    // The SST-2 checkpoint only ships the legacy vocab.txt / tokenizer_config
    // triple, not a fast `tokenizer.json`. Grab the tokenizer from the base
    // repo - the vocabulary is identical since SST-2 fine-tuning does not
    // retrain it.
    let tok_repo = "distilbert-base-uncased";

    // Load pre-trained model. The head has 2 classes:
    // id 0 = NEGATIVE, id 1 = POSITIVE (matches HF convention).
    let head = DistilBertForSequenceClassification::from_pretrained(model_repo)?;
    let tok = HfTokenizer::from_pretrained(tok_repo)?;

    // Tiny domain-specific dataset. Swap this for a real one by
    // feeding a `DataSet` through `DataLoader::new` - the training
    // body below is unchanged.
    let train: &[(&str, i64)] = &[
        ("This framework is a real joy to work with",          1),
        ("I absolutely love the clean API surface",            1),
        ("Releases land on schedule and the diff is readable", 1),
        ("The documentation is thorough and honest",           1),
        ("Fine-tuning just worked on the first try",           1),
        ("The tokenizer is painfully slow",                    0),
        ("I wasted an afternoon chasing a silent shape bug",   0),
        ("The error messages are useless",                     0),
        ("I cannot figure out which feature flag I need",      0),
        ("Performance fell off a cliff after the update",      0),
    ];

    let params = head.graph().parameters();
    let mut opt = Adam::new(&params, 5e-5);
    head.graph().train();

    println!("fine-tuning {model_repo} on {} examples", train.len());
    println!("{:>4} {:>8}", "step", "loss");

    for step in 0..5 {
        opt.zero_grad();

        let texts: Vec<&str> = train.iter().map(|(t, _)| *t).collect();
        let enc = tok.encode(&texts)?;
        let label_ids: Vec<i64> = train.iter().map(|(_, l)| *l).collect();
        let labels = Variable::new(
            Tensor::from_i64(&label_ids, &[train.len() as i64], Device::CPU)?,
            false,
        );

        let loss = head.compute_loss(&enc, &labels)?;
        let loss_val = loss.item()?;
        println!("{:>4} {:>8.4}", step, loss_val);

        loss.backward()?;
        // Standard fine-tuning clip. Keeps early steps stable even
        // when a noisy mini-batch produces a large gradient.
        clip_grad_norm(&params, 1.0)?;
        opt.step()?;
    }

    // Close the loop: run one last eval forward (no backward) and
    // peek at the logits for the first example. POSITIVE logit
    // should sit comfortably above NEGATIVE after the fine-tune.
    head.graph().eval();
    let enc = tok.encode(&[train[0].0])?;
    let logits = head.forward_encoded(&enc)?;
    let flat = logits.data().to_f32_vec()?;
    println!(
        "\nafter fine-tuning, {:?} -> NEGATIVE={:.3}, POSITIVE={:.3}",
        train[0].0, flat[0], flat[1],
    );

    Ok(())
}
