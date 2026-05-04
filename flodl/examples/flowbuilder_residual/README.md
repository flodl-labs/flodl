# FlowBuilder residual

Build a small MLP classifier with a tagged residual block and emit the
SVG used in the FlowBuilder dev.to post.

```sh
cargo run --example flowbuilder_residual
```

Writes `flowbuilder-residual.svg` next to the example.

## What it covers

- `FlowBuilder::from` / `through` / `also` / `tag` / `build`
- `Graph::svg` for static visualization
