//! Graph tree: hierarchical subgraph composition with label-path addressing.
//!
//! When a labeled [`Graph`] is used inside a [`FlowBuilder`](super::FlowBuilder),
//! the parent registers it as a child subgraph. Dot-separated label paths
//! (`"encoder.scan.hidden"`) address subgraphs and tags across boundaries.
//!
//! All operations are build-time or explicit-query-time. The forward path is untouched.
//!
//! # Key methods on [`Graph`]
//!
//! - **Navigation**: [`tree_children()`](Graph::tree_children), [`child_graph()`](Graph::child_graph),
//!   [`subgraph()`](Graph::subgraph), [`is_composed()`](Graph::is_composed)
//! - **Parameters**: [`parameters_at()`](Graph::parameters_at), [`named_parameters_at()`](Graph::named_parameters_at)
//! - **Freeze/thaw**: [`freeze()`](Graph::freeze), [`thaw()`](Graph::thaw), [`is_frozen()`](Graph::is_frozen)
//! - **Checkpoints**: [`load_subgraph_checkpoint()`](Graph::load_subgraph_checkpoint)
//! - **Observation**: [`tagged_at()`](Graph::tagged_at), [`collect_at()`](Graph::collect_at),
//!   [`record_at()`](Graph::record_at), [`trend_at()`](Graph::trend_at)

use std::collections::HashMap;
use crate::autograd::Variable;
use crate::nn::{self, Buffer, Module, Parameter};
use crate::tensor::{Result, TensorError};
use super::Graph;
use super::trend::Trend;

/// What a label path resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathKind {
    /// An entire child subgraph.
    Subgraph,
    /// A named tag within a graph.
    Tag,
}

/// Internal resolution result (borrowed references, no ownership).
#[allow(dead_code)]
pub(crate) enum ResolvedPath<'a> {
    /// Path resolves to an entire child subgraph.
    Subgraph(&'a Graph),
    /// Path resolves to a tag within a specific graph.
    Tag { graph: &'a Graph, tag: String },
}

impl Graph {
    // ── Path resolution ──────────────────────────────────────────────

    /// Resolve a dot-separated label path to a subgraph or tag.
    ///
    /// **Strict dot semantics:**
    /// - `"scan"` — local: check children first (Subgraph), then tags (Tag).
    /// - `"letter.scan"` — child `"letter"`, then `"scan"` within it.
    /// - `"letter.scan.location"` — child `"letter"`, child/tag `"scan"`, then `"location"`.
    ///
    /// Returns `Err` if any segment doesn't resolve.
    pub(crate) fn resolve(&self, path: &str) -> Result<ResolvedPath<'_>> {
        if path.is_empty() {
            return Err(TensorError::new("empty label path"));
        }
        let segments: Vec<&str> = path.split('.').collect();
        self.resolve_segments(&segments, path, false)
    }

    fn resolve_segments<'a>(
        &'a self,
        segments: &[&str],
        full_path: &str,
        cross_boundary: bool,
    ) -> Result<ResolvedPath<'a>> {
        debug_assert!(!segments.is_empty());
        let first = segments[0];

        if segments.len() == 1 {
            // Single segment: children take priority over tags
            if let Some(g) = self.child_graph(first) {
                return Ok(ResolvedPath::Subgraph(g));
            }
            if self.tag_names.contains_key(first) {
                // Block internal tags when accessed from outside
                if cross_boundary && self.internal_tags.contains(first) {
                    return Err(TensorError::new(&format!(
                        "tag {:?} is internal and cannot be accessed from a parent graph (path: {:?})",
                        first, full_path
                    )));
                }
                return Ok(ResolvedPath::Tag { graph: self, tag: first.to_string() });
            }
            return Err(TensorError::new(&format!(
                "{:?} is not a subgraph or tag of this graph (path: {:?})",
                first, full_path
            )));
        }

        // Multi-segment: first MUST be a child label
        let child = self.child_graph(first).ok_or_else(|| {
            TensorError::new(&format!(
                "{:?} is not a subgraph of this graph (path: {:?})",
                first, full_path
            ))
        })?;

        // Once we cross into a child, all subsequent resolution is cross-boundary
        child.resolve_segments(&segments[1..], full_path, true)
    }

    // ── Public navigation ────────────────────────────────────────────

    /// Direct children: label -> child graph.
    pub fn tree_children(&self) -> HashMap<&str, &Graph> {
        self.children.iter()
            .filter_map(|(label, &ni)| {
                self.nodes[ni].module.as_ref()
                    .and_then(|m| m.as_graph())
                    .map(|g| (label.as_str(), g))
            })
            .collect()
    }

    /// Get a direct child graph by label (one level only).
    pub fn child_graph(&self, label: &str) -> Option<&Graph> {
        self.children.get(label)
            .and_then(|&ni| self.nodes[ni].module.as_ref())
            .and_then(|m| m.as_graph())
    }

    /// Get a subgraph at any depth via dot-path.
    pub fn subgraph(&self, path: &str) -> Result<&Graph> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(g) => Ok(g),
            ResolvedPath::Tag { .. } => Err(TensorError::new(&format!(
                "path {:?} resolves to a tag, not a subgraph", path
            ))),
        }
    }

    /// Whether this graph has been composed into a parent graph.
    pub fn is_composed(&self) -> bool {
        self.composed.get()
    }

    /// Tags marked as internal (hidden from parent resolution).
    pub fn internal_tags(&self) -> &std::collections::HashSet<String> {
        &self.internal_tags
    }

    /// Validate that a path resolves, returning what it resolves to.
    pub fn validate_path(&self, path: &str) -> Result<PathKind> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(_) => Ok(PathKind::Subgraph),
            ResolvedPath::Tag { .. } => Ok(PathKind::Tag),
        }
    }

    // ── Parameter operations ─────────────────────────────────────────

    /// All parameters at a label path.
    pub fn parameters_at(&self, path: &str) -> Result<Vec<Parameter>> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(g) => Ok(g.parameters()),
            ResolvedPath::Tag { graph, ref tag } => {
                if let Some(&(ni, _)) = graph.tag_names.get(tag.as_str()) {
                    if let Some(ref module) = graph.nodes[ni].module {
                        Ok(module.parameters())
                    } else {
                        Ok(vec![])
                    }
                } else {
                    Ok(vec![])
                }
            }
        }
    }

    /// Named parameters at a label path, using the target's own namespace.
    /// For subgraphs: delegates to the child graph's `named_parameters()`.
    /// For tags: qualifies with the tag name as prefix.
    pub fn named_parameters_at(&self, path: &str) -> Result<Vec<(String, Parameter)>> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(g) => Ok(g.named_parameters()),
            ResolvedPath::Tag { graph, ref tag } => {
                if let Some(&(ni, _)) = graph.tag_names.get(tag.as_str()) {
                    if let Some(ref module) = graph.nodes[ni].module {
                        Ok(module.parameters().into_iter()
                            .map(|p| (format!("{}/{}", tag, p.name), p))
                            .collect())
                    } else {
                        Ok(vec![])
                    }
                } else {
                    Ok(vec![])
                }
            }
        }
    }

    /// Named buffers at a label path, using the target's own namespace.
    pub fn named_buffers_at(&self, path: &str) -> Result<Vec<(String, Buffer)>> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(g) => Ok(g.named_buffers()),
            ResolvedPath::Tag { graph, ref tag } => {
                if let Some(&(ni, _)) = graph.tag_names.get(tag.as_str()) {
                    if let Some(ref module) = graph.nodes[ni].module {
                        Ok(module.buffers().into_iter()
                            .map(|b| (format!("{}/{}", tag, b.name), b))
                            .collect())
                    } else {
                        Ok(vec![])
                    }
                } else {
                    Ok(vec![])
                }
            }
        }
    }

    // ── Freeze / thaw ────────────────────────────────────────────────

    /// Freeze all parameters at the given label path.
    pub fn freeze(&self, path: &str) -> Result<()> {
        for p in self.parameters_at(path)? {
            p.freeze()?;
        }
        Ok(())
    }

    /// Thaw (unfreeze) all parameters at the given label path.
    pub fn thaw(&self, path: &str) -> Result<()> {
        for p in self.parameters_at(path)? {
            p.unfreeze()?;
        }
        Ok(())
    }

    /// Check if all parameters at the path are frozen.
    /// Returns true only if there are parameters and ALL are frozen.
    pub fn is_frozen(&self, path: &str) -> Result<bool> {
        let params = self.parameters_at(path)?;
        if params.is_empty() {
            return Ok(false);
        }
        Ok(params.iter().all(|p| p.is_frozen()))
    }

    // ── Training mode ────────────────────────────────────────────────

    // ── Checkpoint composition ────────────────────────────────────────

    /// Load a checkpoint into a specific subgraph.
    ///
    /// The checkpoint's structural hash is validated against the target
    /// subgraph's hash. Named parameters/buffers are matched within the
    /// subgraph's own namespace.
    pub fn load_subgraph_checkpoint(&self, path: &str, file: &str) -> Result<nn::LoadReport> {
        let target = self.subgraph(path)?;
        let params = target.named_parameters();
        let buffers = target.named_buffers();
        let hash = target.structural_hash();
        nn::load_checkpoint_file(file, &params, &buffers, Some(hash))
    }

    // ── Training mode ────────────────────────────────────────────────

    /// Set training mode on a specific subgraph or tagged module.
    pub fn set_training_at(&self, path: &str, training: bool) -> Result<()> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(g) => {
                g.set_training(training);
            }
            ResolvedPath::Tag { graph, ref tag } => {
                if let Some(&(ni, _)) = graph.tag_names.get(tag.as_str()) {
                    if let Some(ref module) = graph.nodes[ni].module {
                        crate::nn::walk_modules(module.as_ref(), &mut |m| {
                            m.set_training(training);
                        });
                    }
                }
            }
        }
        Ok(())
    }

    // ── Cross-boundary observation ───────────────────────────────────

    /// Get a tagged output by label path.
    /// Returns `Err` if the path doesn't exist (null -- wiring bug).
    /// Returns `Ok(None)` if the path exists but hasn't been computed yet (nil).
    /// Returns `Ok(Some(v))` if the value is available.
    pub fn tagged_at(&self, path: &str) -> Result<Option<Variable>> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(_) => Err(TensorError::new(&format!(
                "path {:?} resolves to a subgraph, not a tag", path
            ))),
            ResolvedPath::Tag { graph, ref tag } => Ok(graph.tagged(tag)),
        }
    }

    /// Collect metrics from label paths into observation buffers.
    /// Each path must resolve to a tag (not a subgraph).
    /// Metrics are stored in the target graph's batch buffer.
    pub fn collect_at(&self, paths: &[&str]) -> Result<()> {
        for &path in paths {
            match self.resolve(path)? {
                ResolvedPath::Subgraph(_) => {
                    return Err(TensorError::new(&format!(
                        "collect_at: {:?} resolves to a subgraph, not a tag", path
                    )));
                }
                ResolvedPath::Tag { graph, ref tag } => {
                    graph.collect(&[tag.as_str()])?;
                }
            }
        }
        Ok(())
    }

    /// Record a scalar metric at a label path.
    /// For dotted paths, the metric is stored in the target graph's buffer
    /// under the final segment name.
    pub fn record_at(&self, path: &str, value: f64) -> Result<()> {
        let segments: Vec<&str> = path.split('.').collect();
        if segments.len() < 2 {
            // Single segment: record into self
            self.record_scalar(path, value);
            return Ok(());
        }
        // Multi-segment: resolve parent graph, record under last segment
        let parent_path = segments[..segments.len() - 1].join(".");
        let tag = segments[segments.len() - 1];
        let target = self.subgraph(&parent_path)?;
        target.record_scalar(tag, value);
        Ok(())
    }

    /// Get trend for a label-path metric.
    /// For dotted paths, reads from the target graph's epoch history.
    pub fn trend_at(&self, path: &str) -> Result<Trend> {
        let segments: Vec<&str> = path.split('.').collect();
        if segments.len() < 2 {
            return Ok(self.trend(path));
        }
        let parent_path = segments[..segments.len() - 1].join(".");
        let tag = segments[segments.len() - 1];
        let target = self.subgraph(&parent_path)?;
        Ok(target.trend(tag))
    }
}

#[cfg(test)]
mod tests {
    use crate::autograd::Variable;
    use crate::graph::FlowBuilder;
    use crate::nn::{Linear, Module};
    use crate::nn::ReLU;
    use crate::tensor::{test_device, test_opts, Tensor};
    use super::PathKind;

    #[test]
    fn test_unlabeled_graph_no_children() {
        let dev = test_device();

        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .through(ReLU::new())
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        // Unlabeled child is NOT registered
        assert!(outer.tree_children().is_empty());
        // But parameters are still collected (backward compat)
        assert_eq!(outer.parameters().len(), 4); // 2 from inner Linear + 2 from outer Linear
    }

    #[test]
    fn test_labeled_child_registered() {
        let dev = test_device();

        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .through(ReLU::new())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        assert_eq!(outer.tree_children().len(), 1);
        assert!(outer.tree_children().contains_key("encoder"));
        assert!(outer.child_graph("encoder").is_some());
        assert_eq!(outer.child_graph("encoder").unwrap().label(), Some("encoder"));
    }

    #[test]
    fn test_composed_flag() {
        let dev = test_device();

        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("child")
            .build()
            .unwrap();

        // Standalone: not composed
        assert!(!inner.is_composed());

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        // After composition: child is composed
        let child = outer.child_graph("child").unwrap();
        assert!(child.is_composed());
        // Parent is not composed
        assert!(!outer.is_composed());
    }

    #[test]
    fn test_label_collision_error() {
        let dev = test_device();

        let a = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("dupe")
            .build()
            .unwrap();
        let b = FlowBuilder::from(Linear::on_device(4, 2, dev).unwrap())
            .label("dupe")
            .build()
            .unwrap();

        let result = FlowBuilder::from(a)
            .through(b)
            .build();

        let msg = result.err().expect("should be Err").to_string();
        assert!(msg.contains("duplicate child graph label"), "got: {}", msg);
    }

    #[test]
    fn test_dot_in_label_error() {
        let dev = test_device();

        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("a.b")
            .build()
            .unwrap();

        let result = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build();

        let msg = result.err().expect("should be Err").to_string();
        assert!(msg.contains("contains a dot"), "got: {}", msg);
    }

    #[test]
    fn test_label_tag_same_node_ok() {
        let dev = test_device();

        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        // Tag the same node as the child graph label
        let outer = FlowBuilder::from(inner)
            .tag("encoder")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build();

        assert!(outer.is_ok());
    }

    #[test]
    fn test_resolve_single_segment_child() {
        let dev = test_device();

        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        assert_eq!(outer.validate_path("encoder").unwrap(), PathKind::Subgraph);
    }

    #[test]
    fn test_resolve_single_segment_tag() {
        let dev = test_device();

        let outer = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .tag("hidden")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        assert_eq!(outer.validate_path("hidden").unwrap(), PathKind::Tag);
    }

    #[test]
    fn test_resolve_multi_segment() {
        let dev = test_device();

        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .tag("hidden")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        assert_eq!(outer.validate_path("encoder.hidden").unwrap(), PathKind::Tag);
    }

    #[test]
    fn test_resolve_multi_level() {
        let dev = test_device();

        let innermost = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("read")
            .build()
            .unwrap();
        let middle = FlowBuilder::from(innermost)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("letter")
            .build()
            .unwrap();
        let outer = FlowBuilder::from(middle)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        assert_eq!(outer.validate_path("letter").unwrap(), PathKind::Subgraph);
        assert_eq!(outer.validate_path("letter.read").unwrap(), PathKind::Subgraph);
    }

    #[test]
    fn test_resolve_invalid_path_error() {
        let dev = test_device();

        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        // Non-existent single segment
        assert!(outer.validate_path("nonexistent").is_err());
        // Non-existent dotted path
        assert!(outer.validate_path("encoder.nonexistent").is_err());
        // Dotting into non-child first segment
        assert!(outer.validate_path("nonexistent.foo").is_err());
    }

    #[test]
    fn test_subgraph_returns_graph() {
        let dev = test_device();

        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        let sub = outer.subgraph("encoder").unwrap();
        assert_eq!(sub.label(), Some("encoder"));
        assert_eq!(sub.parameters().len(), 2); // 1 Linear: weight + bias
    }

    #[test]
    fn test_forward_still_works_with_tree() {
        let dev = test_device();
        let opts = test_opts();

        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .through(ReLU::new())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(
            Tensor::randn(&[1, 3], opts).unwrap(),
            false,
        );
        let y = outer.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    // ── Phase B: Training control ────────────────────────────────────

    #[test]
    fn test_parameters_at_subgraph() {
        let dev = test_device();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        // Child has 2 Linear layers = 4 params (2 weight + 2 bias)
        let params = outer.parameters_at("encoder").unwrap();
        assert_eq!(params.len(), 4);
        // Outer total = 4 (child) + 2 (outer Linear) = 6
        assert_eq!(outer.parameters().len(), 6);
    }

    #[test]
    fn test_parameters_at_tag() {
        let dev = test_device();
        let g = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .tag("first")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        let params = g.parameters_at("first").unwrap();
        assert_eq!(params.len(), 2); // 1 Linear: weight + bias
    }

    #[test]
    fn test_freeze_thaw_roundtrip() {
        let dev = test_device();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        // Initially not frozen
        assert!(!outer.is_frozen("encoder").unwrap());

        // Freeze child
        outer.freeze("encoder").unwrap();
        assert!(outer.is_frozen("encoder").unwrap());
        // All child params should have requires_grad = false
        for p in outer.parameters_at("encoder").unwrap() {
            assert!(p.is_frozen());
        }
        // Outer params still trainable
        let outer_params = outer.parameters();
        let outer_only: Vec<_> = outer_params.iter()
            .filter(|p| !p.is_frozen())
            .collect();
        assert_eq!(outer_only.len(), 2); // outer Linear: weight + bias

        // Thaw child
        outer.thaw("encoder").unwrap();
        assert!(!outer.is_frozen("encoder").unwrap());
        for p in outer.parameters_at("encoder").unwrap() {
            assert!(!p.is_frozen());
        }
    }

    #[test]
    fn test_freeze_deep_path() {
        let dev = test_device();
        let innermost = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("read")
            .build()
            .unwrap();
        let middle = FlowBuilder::from(innermost)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("letter")
            .build()
            .unwrap();
        let outer = FlowBuilder::from(middle)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        // Freeze only the innermost
        outer.freeze("letter.read").unwrap();
        assert!(outer.is_frozen("letter.read").unwrap());
        // "letter" overall is NOT fully frozen (it has its own Linear too)
        assert!(!outer.is_frozen("letter").unwrap());
    }

    #[test]
    fn test_named_parameters_at_uses_target_namespace() {
        let dev = test_device();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .tag("hidden")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        // Subgraph: uses child's own namespace
        let named = outer.named_parameters_at("encoder").unwrap();
        assert_eq!(named.len(), 4);
        // Names should use child-local prefixes (tag "hidden" and node id)
        assert!(named.iter().any(|(n, _)| n.starts_with("hidden/")));
    }

    #[test]
    fn test_freeze_invalid_path_error() {
        let dev = test_device();
        let g = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .build()
            .unwrap();

        assert!(g.freeze("nonexistent").is_err());
        assert!(g.thaw("nonexistent").is_err());
        assert!(g.is_frozen("nonexistent").is_err());
        assert!(g.parameters_at("nonexistent").is_err());
    }

    #[test]
    fn test_set_training_at() {
        let dev = test_device();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .through(crate::nn::Dropout::new(0.5))
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        // Set child to eval mode
        outer.set_training_at("encoder", false).unwrap();
        // Set child back to training mode
        outer.set_training_at("encoder", true).unwrap();
        // Invalid path errors
        assert!(outer.set_training_at("nonexistent", false).is_err());
    }

    // ── Phase C: Checkpoint composition ──────────────────────────────

    #[test]
    fn test_subgraph_checkpoint_roundtrip() {
        let dev = test_device();
        // Build and "train" a child graph standalone
        let child = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        // Save child checkpoint
        let dir = std::env::temp_dir().join("flodl_test_subgraph_ckpt");
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt_path = dir.join("encoder.fdl");
        child.save_checkpoint(ckpt_path.to_str().unwrap()).unwrap();

        // Build parent with a fresh (randomly initialized) child of same architecture
        let fresh_child = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let parent = FlowBuilder::from(fresh_child)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        // Load child checkpoint into parent's subgraph
        let report = parent.load_subgraph_checkpoint("encoder", ckpt_path.to_str().unwrap()).unwrap();
        assert!(report.loaded.len() >= 4); // At least weight+bias from 2 Linears
        assert!(report.missing.is_empty());

        // Clean up
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_subgraph_checkpoint_preserves_parent_params() {
        let dev = test_device();
        let child = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let dir = std::env::temp_dir().join("flodl_test_preserve_parent");
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt_path = dir.join("encoder.fdl");
        child.save_checkpoint(ckpt_path.to_str().unwrap()).unwrap();

        // Build parent with fresh child
        let fresh_child = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();
        let parent = FlowBuilder::from(fresh_child)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        // Snapshot parent-level param data
        let parent_w = parent.parameters().last().unwrap().variable.data().clone();

        // Load child checkpoint
        parent.load_subgraph_checkpoint("encoder", ckpt_path.to_str().unwrap()).unwrap();

        // Parent param unchanged
        let parent_w_after = parent.parameters().last().unwrap().variable.data().clone();
        let diff = parent_w.sub(&parent_w_after).unwrap().abs().unwrap().sum().unwrap().item().unwrap();
        assert!(diff < 1e-10, "parent params should be unchanged, diff={}", diff);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Phase D: Cross-boundary observation ──────────────────────────

    #[test]
    fn test_tagged_at_returns_value_after_forward() {
        let dev = test_device();
        let opts = test_opts();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .tag("hidden")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(Tensor::randn(&[1, 3], opts).unwrap(), false);
        outer.forward(&x).unwrap();

        let val = outer.tagged_at("encoder.hidden").unwrap();
        assert!(val.is_some());
        assert_eq!(val.unwrap().shape(), vec![1, 4]);
    }

    #[test]
    fn test_tagged_at_before_forward_returns_none() {
        let dev = test_device();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .tag("hidden")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        // Before forward: path exists but no value computed
        let val = outer.tagged_at("encoder.hidden").unwrap();
        assert!(val.is_none());
    }

    #[test]
    fn test_tagged_at_invalid_path_returns_err() {
        let dev = test_device();
        let g = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .build()
            .unwrap();

        assert!(g.tagged_at("nonexistent.tag").is_err());
    }

    #[test]
    fn test_record_at_and_trend_at() {
        let dev = test_device();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        // Record into child's buffer
        outer.record_at("encoder.loss", 0.5).unwrap();
        outer.record_at("encoder.loss", 0.3).unwrap();

        // Flush child's buffers to see the trend
        let child = outer.child_graph("encoder").unwrap();
        child.flush(&[]);

        let trend = outer.trend_at("encoder.loss").unwrap();
        assert_eq!(trend.len(), 1); // one epoch flushed
    }

    // ── Phase E: Developer experience ────────────────────────────────

    #[test]
    fn test_internal_tag_hidden_from_parent() {
        let dev = test_device();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .tag("_plumbing")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .tag("output")
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        // Auto-internal: _plumbing starts with underscore
        assert!(outer.child_graph("encoder").unwrap().internal_tags().contains("_plumbing"));
        // Internal tag blocked from parent
        assert!(outer.tagged_at("encoder._plumbing").is_err());
        // Non-internal tag accessible
        assert_eq!(outer.validate_path("encoder.output").unwrap(), PathKind::Tag);
    }

    #[test]
    fn test_explicit_internal_tag() {
        let dev = test_device();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .tag("intermediate")
            .internal("intermediate")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        // Explicitly internal: blocked from parent
        assert!(outer.tagged_at("encoder.intermediate").is_err());
    }

    #[test]
    fn test_tree_summary_output() {
        let dev = test_device();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .tag("hidden")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();

        let summary = outer.tree_summary();
        assert!(summary.contains("Graph Tree"), "missing header:\n{}", summary);
        assert!(summary.contains("encoder"), "missing child label:\n{}", summary);
        assert!(summary.contains("Parameter Summary"), "missing param summary:\n{}", summary);
    }

    #[test]
    fn test_param_summary_output() {
        let dev = test_device();
        let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
            .label("encoder")
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        let summary = outer.param_summary();
        assert!(summary.contains("encoder"), "missing child:\n{}", summary);
        assert!(summary.contains("(own)"), "missing own params:\n{}", summary);
        assert!(summary.contains("trainable"), "missing trainable:\n{}", summary);
    }
}
