# Showcase

Exercises every method of the graph fluent builder API in a single coherent
graph, plus training tools, observation, and visualization.

```sh
cargo run --example showcase
```

## Builder methods exercised

`from`, `through`, `tag`, `using` (backward & forward refs), `split`, `merge`,
`also`, `map` (slices/each/over/batched), `loop_body` (for/while/until),
`gate`, `switch`, `fork`, `input`, `tag_group`, `build`

## Graph methods exercised

`forward`, `forward_multi`, `parameters`, `set_training`, `reset_state`,
`detach_state`, `dot`, `tagged`, `flush`, `trend`, `enable_profiling`,
`flush_timings`, `timing_trend`

## Training tools exercised

Adam optimizer, CosineScheduler, mse_loss, clip_grad_norm,
save/load checkpoint, no_grad, observation, profiling, trends

## Generated artifacts

| File | Description |
|------|-------------|
| `showcase.dot` | Structural graph in DOT format |
| `showcase.svg` | Structural graph rendered as SVG |
| `showcase_profile.dot` | Graph with profiling heat map (DOT) |
| `showcase_profile.svg` | Graph with profiling heat map (SVG) |
| `showcase_training.html` | Interactive training curves |
| `showcase_training.log` | Training log (epoch summaries) |
