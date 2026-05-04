//! flowbuilder_residual — illustrative graph for the FlowBuilder dev.to post.
//!
//! Builds a small MLP classifier with a labeled encoder subgraph that
//! contains a tagged residual block, then writes the SVG referenced in
//! the post: `site/assets/images/flowbuilder-residual.svg`.
//!
//! Run: `cargo run --example flowbuilder_residual`

use flodl::*;

fn main() -> Result<()> {
    let model = FlowBuilder::from(Linear::new(784, 128)?)
        .through(GELU)
        .tag("hidden")
        .through(LayerNorm::new(128)?)
        .also(Linear::new(128, 128)?)
        .tag("residual")
        .through(Linear::new(128, 10)?)
        .build()?;

    let svg_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/flowbuilder_residual/flowbuilder-residual.svg"
    );
    let svg = model.svg(Some(svg_path))?;
    println!("Wrote {} ({} bytes)", svg_path, svg.len());

    Ok(())
}
