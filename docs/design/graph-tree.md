# Graph Tree: Composable Subgraph Training

A design for hierarchical graph composition with cross-boundary tag resolution,
label-path addressing, and per-subgraph training control.

This feature enables the "hierarchical composition" and "multi-strategy training"
described in the trajectory thesis — trained graphs nested as modules inside
parent graphs, with fine-grained control over which parts train, freeze, or
share gradients.

---

## Motivation

flodl already supports Graph-as-Module composition: a built `Graph` implements
`Module` and can be placed inside another `FlowBuilder`. But training a composed
system requires capabilities that don't yet exist:

1. **Selective freezing** — freeze the read phase of a letter model while its
   scan phase adapts to word-level context.
2. **Cross-boundary observation** — a parent graph needs to read a child's
   tagged outputs (e.g., confidence scores) to make routing decisions.
3. **Subgraph checkpointing** — load a pre-trained letter model into a subtree
   of a larger word model, without disturbing surrounding weights.
4. **Per-subgraph optimizer groups** — different learning rates for different
   subgraphs, collected by label path rather than manual parameter wrangling.
5. **Training phase control** — declarative freeze/thaw schedules expressed as
   label paths, not imperative `requires_grad` surgery.

### Driving use case: hierarchical word recognition

```
WordGraph "word"
├── MetaScan "meta"        — coarse sweep, finds word boundaries
├── SubScan "subscan"      — proposes letter positions within word
└── LetterGraph "letter"   — pre-trained, composed as subgraph
    ├── ScanLoop "scan"    — learnable (adapts to word context)
    └── ReadLoop "read"    — frozen (proven 100% on letter data)
```

The word graph needs to:
- Load a letter checkpoint into `"letter"` without touching `"meta"` or `"subscan"`
- Freeze `"letter.read"` while training everything else
- Read `"letter.read.confidence"` to decide whether to retry with an adjusted position
- Assign different LRs: `"meta"` and `"subscan"` at 0.001, `"letter.scan"` at 0.0001

None of this is possible today without manual parameter bookkeeping.

---

## Design

### 1. Graph tree structure

Every `Graph` gains optional parent/child relationships:

```rust
pub struct Graph {
    // ... existing fields ...

    /// Human-readable label (set via FlowBuilder::label).
    /// Stable across development — "letter" stays "letter" even if
    /// internal modules change (which would change the structural hash).
    label: Option<String>,

    /// Child subgraphs, keyed by label.
    children: HashMap<String, Rc<Graph>>,

    /// Weak reference to parent (None for root graph).
    parent: Option<Weak<Graph>>,
}
```

Children are registered automatically when a `Graph` (with a label) is used as
a `Module` inside a `FlowBuilder`. The builder detects that the module is a
`Graph`, reads its label, and registers the parent/child relationship.

**Build-time validation:**
- Label collision at the same tree level is a build error.
- A graph used as a child subgraph *must* have a label (builder error otherwise).
- Labels must be valid identifiers (alphanumeric + underscore, no dots).

### 2. Label-path addressing

All tree operations use dot-separated label paths with optional loop indexing:

```
"letter"                    → entire letter subgraph
"letter.scan"               → scan tag/subgraph within letter
"letter.scan[3]"            → scan loop iteration 3
"letter.scan[3].location"   → tag "location" within scan step 3
"meta"                      → meta-scan subgraph
"subscan.hit_rate"          → hit_rate tag within subscan
```

**Resolution rules** (closest scope outward):
- Bare names (e.g., `"scan"`) resolve from the current graph first, then walk
  up to parent, then down to children.
- If a bare name is ambiguous (exists in multiple children), it is a build-time
  error — the user must qualify: `"letter.scan"` vs `"meta.scan"`.
- Qualified paths (containing dots) resolve by walking the tree from the root
  of the path: `"letter.scan"` means "child labeled `letter`, then tag/child
  labeled `scan` within it."

**Loop indexing:**
- `"scan"` resolves to the loop's collected output (all iterations).
- `"scan[i]"` resolves to iteration `i`'s output specifically.
- Negative indexing: `"scan[-1]"` for last iteration.
- Range: `"scan[2..5]"` for iterations 2, 3, 4 (future, if needed).

This extends the existing `tag_names` and `tag_groups` system. Currently tags
are flat within a graph; this adds hierarchical resolution across the tree.

### 3. Tag visibility: shared by default, opt-in internal

**Design principle:** tags exist because someone intended reuse. Respect that
intent. Default is visible across all graph boundaries.

Every tag in every subgraph is accessible from any point in the tree via its
label path. No port declarations, no explicit exports.

**Opt-in internal:** for tags that are truly implementation details:

```rust
FlowBuilder::from(module)
    .through(intermediate)
    .tag("_plumbing")           // convention: underscore prefix = internal
    .internal("_plumbing")      // explicit: hide from parent/sibling resolution
    .through(next)
    .tag("output")              // visible everywhere
    .build()
```

Internal tags:
- Cannot be resolved from outside their graph.
- Do not appear in verbose wiring output.
- Do not contribute to label-path ambiguity checks.

**Auto-internal inference:** some tags are obviously internal and should be
marked automatically, without requiring the developer to think about it:

| Pattern | Reason | Auto-internal? |
|---|---|---|
| `.using("ref")` forward-ref wiring | Plumbing between modules | Yes |
| Loop iteration counter | Bookkeeping | Yes |
| Intermediate concat/reshape | Structural glue | Yes |
| Tags starting with `_` | Convention | Yes |
| `.tag("loss")`, `.tag("location")` | Semantic output | No |
| Loop trace outputs | Diagnostic/reuse | No |

Auto-internal is a heuristic. If the developer disagrees, they can explicitly
mark a tag as shared (`tag("_name").shared()`) to override.

**Build-time warning:** if a non-internal tag in a child graph is never
referenced by any parent or sibling, emit a warning: "tag `foo` in `letter`
is never referenced externally — consider marking internal." This keeps the
namespace clean without breaking anything.

### 4. Tree-aware parameter collection

```rust
impl Graph {
    /// Parameters matching a label path.
    /// "letter.scan" → all parameters in the scan phase of the letter subgraph.
    /// "letter" → all parameters in the entire letter subgraph.
    /// "" or no arg → all parameters in the entire tree (existing behavior).
    pub fn parameters_at(&self, path: &str) -> Vec<Parameter> { ... }
}
```

This walks the tree to the target node, then collects parameters from that
subtree. Deduplication by pointer (existing behavior) prevents double-counting
shared parameters.

The existing `parameters()` method remains unchanged (returns all parameters
in the graph, including children — backward compatible).

### 5. Tree-aware freeze/thaw

```rust
impl Graph {
    /// Freeze all parameters at the given label path.
    /// Sets requires_grad = false on all parameters in the subtree.
    pub fn freeze(&self, path: &str) { ... }

    /// Thaw (unfreeze) all parameters at the given label path.
    pub fn thaw(&self, path: &str) { ... }

    /// Check if a label path is fully frozen.
    pub fn is_frozen(&self, path: &str) -> bool { ... }
}
```

These delegate to `Parameter::freeze()` / `Parameter::unfreeze()` which already
exist. The new part is label-path resolution to find which parameters to target.

**Training phase definitions become declarative:**

```rust
// Phase A: train subscan only, letter frozen
graph.freeze("letter");

// Phase B: thaw letter scan, read stays frozen
graph.thaw("letter.scan");

// Phase C: fine-tune everything except read
graph.thaw("meta");
graph.thaw("subscan");
// "letter.read" remains frozen from Phase A
```

### 6. Subgraph checkpoint loading

```rust
impl Graph {
    /// Load a checkpoint into a specific subgraph.
    ///
    /// The checkpoint's structural hash is validated against the target
    /// subgraph's hash (not the root graph's hash). Named parameters are
    /// matched within the subgraph's namespace.
    ///
    /// Returns LoadReport with loaded/skipped/missing parameter names.
    pub fn load_subgraph_checkpoint(
        &self,
        path: &str,          // label path to target subgraph
        checkpoint: &str,    // file path to checkpoint
    ) -> Result<LoadReport> { ... }
}
```

This is the key to transfer learning in composed systems:

```rust
// Build word model with empty letter subgraph
let word = FlowBuilder::from(Identity)
    .through(meta_scan).tag("meta")
    .through(sub_scan).tag("subscan")
    .through(letter_graph)  // labeled "letter" internally
    .build()?;

// Load pre-trained letter weights into the subgraph
let report = word.load_subgraph_checkpoint("letter", "letter_v2_best.fdl.gz")?;
eprintln!("Loaded {}/{} params into letter subgraph", report.loaded, report.total);

// Freeze the read phase (pre-trained, proven)
word.freeze("letter.read");
```

The structural hash check ensures the letter checkpoint matches the letter
subgraph's architecture, regardless of what the parent graph looks like.

### 7. Cross-boundary tag access

```rust
impl Graph {
    /// Get a tagged output by label path.
    /// Works across graph boundaries — can reach into child subgraphs.
    pub fn tagged_at(&self, path: &str) -> Option<Variable> { ... }

    /// Collect metrics from a label path into observation buffers.
    pub fn collect_at(&self, path: &str) -> Result<()> { ... }

    /// Record a scalar metric at a label path.
    pub fn record_at(&self, path: &str, value: f64) { ... }

    /// Get trend for a label-path metric.
    pub fn trend_at(&self, path: &str) -> Option<Trend> { ... }
}
```

The existing `tagged()`, `collect()`, `record_scalar()`, `trend()` methods
remain unchanged for local (non-hierarchical) access. The `_at` variants add
tree-aware resolution.

**Use case — conditional retry in word model:**

```rust
// After letter model forward pass:
let confidence = word.tagged_at("letter.read.confidence")
    .map(|v| v.item().unwrap_or(0.0))
    .unwrap_or(0.0);

if confidence < threshold {
    // Adjust subscan position and re-dispatch
    let delta = refine_head.forward(&word.tagged_at("letter.read.hidden").unwrap())?;
    // ... retry logic ...
}
```

### 8. Verbose build-time wiring output

A diagnostic mode that prints the complete graph tree, tag resolution map,
and wiring at build time:

```rust
let graph = FlowBuilder::from(Identity)
    // ... build graph ...
    .verbose(true)   // enable build-time diagnostics
    .build()?;
```

Output format:

```
=== Graph Tree ===
word [hash: a3f8c2d1]
├── meta [hash: 7b2e9f4a]
│   ├── tags: location, content_logit
│   └── params: 12,544 (3 modules)
├── subscan [hash: 1c5d8e3b]
│   ├── tags: positions, hit_rate
│   └── params: 98,816 (5 modules)
└── letter [hash: 9e4f2a7c]    ← matches checkpoint letter_v2_best.fdl.gz
    ├── scan [hash: 4d1a8c5e]
    │   ├── tags: location, content_logit
    │   └── params: 45,312 (2 modules)
    └── read [hash: 6f3b9d2a]  ★ frozen
        ├── tags: hidden, confidence, classification
        └── params: 45,312 (2 modules) [frozen]

=== Tag Resolution ===
"location"          → AMBIGUOUS (meta.location, letter.scan.location) — qualify to resolve
"meta.location"     → meta/node_3:0
"positions"         → subscan/node_2:0
"letter.scan[0]"    → letter/scan/loop_body:iter_0
"confidence"        → letter.read.confidence → letter/read/node_5:0

=== Wiring ===
meta.location        → subscan (input)
subscan.positions[k] → letter.scan (start_pos)      [for each k in 0..N]
letter.read.hidden   → refine_head (input)           [conditional retry path]

=== Internal (auto-detected) ===
meta._using_image         auto-internal (forward ref wiring)
letter.scan._step_idx     auto-internal (loop counter)
subscan._concat_tmp       auto-internal (intermediate reshape)

=== Parameter Summary ===
Total: 201,984 parameters
  meta:          12,544  (6.2%)
  subscan:       98,816  (48.9%)
  letter.scan:   45,312  (22.4%)  trainable
  letter.read:   45,312  (22.4%)  frozen
```

This output is only produced when `.verbose(true)` is set. It helps developers
verify that:
- Tags resolve where expected
- Wiring crosses boundaries correctly
- Freeze/thaw state is correct
- Parameter counts match expectations
- No unintended ambiguities exist

### 9. Optimizer integration

The existing `Adam::with_groups()` builder works with label-path parameter
collection:

```rust
let mut optimizer = Adam::with_groups()
    .group(&graph.parameters_at("meta"), 0.001)
    .group(&graph.parameters_at("subscan"), 0.001)
    .group(&graph.parameters_at("letter.scan"), 0.0001)
    // letter.read is frozen — not in any group
    .build();
```

No changes needed to the optimizer itself. The new capability is
`parameters_at()` which makes group construction declarative.

### 10. Training mode propagation

`set_training(bool)` already propagates to child modules. Since `Graph`
implements `Module`, calling `word_graph.train()` sets training mode on all
subgraphs recursively. For selective eval (e.g., letter read in eval mode
during word training):

```rust
impl Graph {
    /// Set training mode on a specific subgraph.
    pub fn set_training_at(&self, path: &str, training: bool) { ... }
}
```

This matters for BatchNorm — frozen subgraphs should use running stats (eval
mode), not batch stats (training mode).

---

## Implementation plan

### Step 1: Graph tree registration

**Files:** `graph/mod.rs`, `graph/flow.rs`

- Add `children: HashMap<String, Rc<Graph>>` and `parent: Option<Weak<Graph>>`
  to `Graph`.
- In `FlowBuilder`, when `.through(module)` receives a `Module` whose
  `structural_hash()` is `Some(_)` (i.e., it's a `Graph`), check for a label
  and register the parent/child relationship.
- Implement label collision detection in `.build()`.
- Require labels on subgraphs (build error if a Graph-as-Module has no label).

**Note on detection:** The builder currently stores modules as `Rc<dyn Module>`.
To detect that a module is a `Graph`, we need either:
- A method on `Module`: `fn as_graph(&self) -> Option<&Graph>` (downcast).
- Or check `structural_hash().is_some()` as a proxy (only Graph has this today).

The cleanest approach is adding `as_graph()` to the Module trait with a default
`None` implementation, overridden in Graph.

### Step 2: Label-path resolution

**Files:** `graph/mod.rs` (new submodule: `graph/tree.rs`)

- Implement path parsing: split on `.`, handle `[i]` indexing.
- Implement resolution: walk tree from current graph, matching path segments
  against child labels and tag names.
- Implement ambiguity detection for bare names.
- Add `resolve(&self, path: &str) -> Result<ResolvedPath>` to `Graph`.

`ResolvedPath` carries enough info to reach the target:

```rust
pub enum ResolvedPath {
    /// A tag within a specific graph.
    Tag { graph: Rc<Graph>, tag: String, index: Option<usize> },
    /// An entire subgraph.
    Subgraph { graph: Rc<Graph> },
}
```

### Step 3: Tag visibility and internal marking

**Files:** `graph/flow.rs`, `graph/tree.rs`

- Add `internal_tags: HashSet<String>` to `Graph`.
- Add `.internal(tag)` to `FlowBuilder`.
- Implement auto-internal inference in `.build()`:
  - Forward-ref wiring targets → internal
  - Tags starting with `_` → internal
  - Loop iteration counters → internal
- Internal tags are excluded from cross-boundary resolution.

### Step 4: Tree-aware parameter operations

**Files:** `graph/mod.rs`

- Implement `parameters_at(&self, path: &str) -> Vec<Parameter>`.
- Implement `freeze(&self, path: &str)` and `thaw(&self, path: &str)`.
- Implement `is_frozen(&self, path: &str) -> bool`.

These resolve the path, then operate on the target subtree's parameters.

### Step 5: Subgraph checkpoint loading

**Files:** `nn/checkpoint.rs`, `graph/mod.rs`

- Implement `load_subgraph_checkpoint(&self, path, file)`.
- Resolve path to target subgraph.
- Validate checkpoint's structural hash against target subgraph's hash.
- Load named parameters/buffers into the subgraph's namespace.
- Return `LoadReport`.

### Step 6: Cross-boundary observation

**Files:** `graph/observe.rs`, `graph/mod.rs`

- Implement `tagged_at(&self, path: &str) -> Option<Variable>`.
- Implement `collect_at()`, `record_at()`, `trend_at()`.
- These resolve the path, then delegate to the target graph's existing
  observation methods.

### Step 7: Verbose build output

**Files:** `graph/flow.rs` (new: `graph/verbose.rs`)

- Add `.verbose(bool)` to `FlowBuilder`.
- On `.build()`, if verbose, walk the tree and print:
  - Tree structure with labels, hashes, param counts
  - Tag resolution map (all resolvable paths)
  - Wiring (cross-boundary connections)
  - Internal tags (auto-detected)
  - Parameter summary with frozen state

### Step 8: Training mode propagation

**Files:** `graph/mod.rs`

- Implement `set_training_at(&self, path, bool)`.
- Resolve path, call `set_training()` on target subgraph.

---

## API summary

New methods on `Graph`:

```rust
// Tree navigation
fn children(&self) -> &HashMap<String, Rc<Graph>>;
fn parent(&self) -> Option<Rc<Graph>>;
fn root(&self) -> Rc<Graph>;
fn subgraph(&self, path: &str) -> Option<Rc<Graph>>;

// Label-path parameter operations
fn parameters_at(&self, path: &str) -> Vec<Parameter>;
fn freeze(&self, path: &str);
fn thaw(&self, path: &str);
fn is_frozen(&self, path: &str) -> bool;

// Label-path observation
fn tagged_at(&self, path: &str) -> Option<Variable>;
fn collect_at(&self, path: &str) -> Result<()>;
fn record_at(&self, path: &str, value: f64);
fn trend_at(&self, path: &str) -> Option<Trend>;

// Subgraph checkpoint
fn load_subgraph_checkpoint(&self, path: &str, file: &str) -> Result<LoadReport>;

// Training mode
fn set_training_at(&self, path: &str, training: bool);
```

New methods on `FlowBuilder`:

```rust
fn internal(&mut self, tag: &str) -> &mut Self;
fn verbose(&mut self, enabled: bool) -> &mut Self;
```

New method on `Module` trait:

```rust
fn as_graph(&self) -> Option<&Graph> { None }
```

---

## Backward compatibility

All existing APIs remain unchanged:
- `parameters()` returns all parameters (now including children).
- `tagged()`, `collect()`, `trend()` work as before for local tags.
- `save_checkpoint()` / `load_checkpoint()` operate on the full graph.
- Tags without dots resolve locally first (existing behavior).

The new `_at` methods and tree operations are purely additive.

---

## Connection to trajectory thesis

This design directly enables the "Mixture of Strategies" and "Hierarchical
Composition" sections of the trajectory thesis:

> Each sub-graph carries its training context (optimizer, loss, schedule).
> Gradient flow between sub-graphs is declared in the graph topology.
> The meta-controller's branching and looping are native Rust — zero overhead.
> Training the meta-controller is just another backward pass through a graph
> that happens to contain other trained graphs as nodes.

With graph tree + label-path addressing:
- "Each sub-graph carries its training context" = `parameters_at()` + per-group optimizer
- "Gradient flow declared in topology" = shared-by-default tags + `freeze()`/`thaw()`
- "Trained graphs as nodes" = subgraph checkpoint loading + selective freezing

The verbose build output makes the composed system inspectable — critical for
debugging multi-strategy training where gradient flow across boundaries is
the most common source of bugs.

---

## Open questions

1. **Shared-weight subgraphs.** If the same `LetterGraph` is used at 4 word
   positions (shared weights, called in a loop), it's one tree node called
   N times. If we want independent weights per position, it's N tree nodes
   (`"letter_0"` through `"letter_3"`). Both patterns should work. The loop
   case is already handled by flodl's `Loop`; the independent case just needs
   distinct labels.

2. **Cross-graph gradient blocking.** Currently, `freeze()` sets
   `requires_grad = false`, which blocks gradients through frozen parameters.
   But what about blocking gradients *between* subgraphs while keeping both
   trainable? E.g., "train letter.scan and subscan independently, don't let
   letter.scan's gradients flow back into subscan." This would need a
   `detach_at()` or gradient barrier at a label path boundary. Not needed for
   the initial use case but worth considering.

3. **Dynamic tree modification.** Can you add/remove subgraphs after build?
   Initial answer: no. The tree is fixed at build time. If you need different
   compositions, build different graphs. This keeps the implementation simple
   and the structural hash meaningful.

4. **Monitoring integration.** The Monitor dashboard currently shows one graph's
   metrics. With a graph tree, the dashboard could show a tree view with
   per-subgraph metrics, expandable/collapsible. This is a Monitor feature,
   not a Graph feature, but worth designing alongside.
