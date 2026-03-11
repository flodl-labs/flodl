use crate::autograd::Variable;

use super::trend::{Trend, TrendGroup};
use super::Graph;

impl Graph {
    /// Get the output of a tagged node from the last forward pass.
    pub fn tagged(&self, tag: &str) -> Option<Variable> {
        self.tagged_outputs.borrow().get(tag).cloned()
    }

    /// Get all tag names defined in this graph.
    pub fn tag_names(&self) -> Vec<String> {
        self.tag_names.keys().cloned().collect()
    }

    /// Snapshot current scalar values of tagged nodes into the batch buffer.
    /// Call once per batch during training.
    pub fn collect(&self, tags: &[&str]) {
        let tagged = self.tagged_outputs.borrow();
        let mut buffer = self.batch_buffer.borrow_mut();
        for &tag in tags {
            if let Some(var) = tagged.get(tag)
                && let Ok(val) = var.item()
            {
                buffer.entry(tag.to_string()).or_default().push(val);
            }
        }
    }

    /// Inject external scalar values into the batch buffer.
    /// Useful for recording metrics not captured by tagged nodes.
    pub fn record(&self, tag: &str, values: &[f64]) {
        let mut buffer = self.batch_buffer.borrow_mut();
        buffer.entry(tag.to_string()).or_default().extend_from_slice(values);
    }

    /// Read raw batch buffer for a tag (all values since last flush).
    pub fn collected(&self, tag: &str) -> Vec<f64> {
        self.batch_buffer.borrow().get(tag).cloned().unwrap_or_default()
    }

    /// Compute batch means, append to epoch history, clear batch buffer.
    /// Call once per epoch. If tags is empty, flushes all buffered tags.
    pub fn flush(&self, tags: &[&str]) {
        let mut buffer = self.batch_buffer.borrow_mut();
        let mut history = self.epoch_history.borrow_mut();

        let keys: Vec<String> = if tags.is_empty() {
            buffer.keys().cloned().collect()
        } else {
            tags.iter().map(|t| t.to_string()).collect()
        };

        let mut flushed_any = false;
        for key in &keys {
            if let Some(values) = buffer.remove(key)
                && !values.is_empty()
            {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                history.entry(key.clone()).or_default().push(mean);
                flushed_any = true;
            }
        }

        if flushed_any {
            let count = self.flush_count.get();
            self.flush_count.set(count + 1);
            self.flush_times.borrow_mut().push(
                super::instant_secs() - self.training_start.get(),
            );
        }
    }

    /// Number of flush calls that produced data.
    pub fn flush_count(&self) -> usize {
        self.flush_count.get()
    }

    /// Get epoch-level trend for a tag.
    pub fn trend(&self, tag: &str) -> Trend {
        let history = self.epoch_history.borrow();
        Trend::new(history.get(tag).cloned().unwrap_or_default())
    }

    /// Get trends for multiple tags. Tag group names are automatically
    /// expanded to their member tags.
    pub fn trends(&self, tags: &[&str]) -> TrendGroup {
        let expanded = self.expand_groups(tags);
        let history = self.epoch_history.borrow();
        let trends = expanded
            .iter()
            .map(|tag| Trend::new(history.get(tag).cloned().unwrap_or_default()))
            .collect();
        TrendGroup(trends)
    }

    /// Clear epoch history. If tags is empty, clears all.
    /// Tag group names are automatically expanded.
    pub fn reset_trend(&self, tags: &[&str]) {
        let mut history = self.epoch_history.borrow_mut();
        if tags.is_empty() {
            history.clear();
        } else {
            let expanded = self.expand_groups(tags);
            for tag in &expanded {
                history.remove(tag);
            }
        }
    }

    /// Get per-iteration trace outputs from loop nodes.
    /// Returns the trace buffer for the loop node associated with the given tag.
    /// The tag should be set on a node after the loop (the loop output flows to it).
    /// Returns None if no loop node with a trace buffer is found.
    pub fn traces(&self, tag: &str) -> Option<Vec<Variable>> {
        // Look for loop nodes by checking trace_buf
        // If a tag is given, find the node it references and walk back to find the loop
        if let Some(&(ni, _)) = self.tag_names.get(tag) {
            // Check if this node has a trace_buf
            if let Some(ref buf) = self.nodes[ni].trace_buf {
                let traces = buf.borrow().clone();
                if !traces.is_empty() {
                    return Some(traces);
                }
            }
        }
        // Search all nodes for a matching tag in the node id
        for node in &self.nodes {
            if let Some(ref buf) = node.trace_buf {
                let traces = buf.borrow().clone();
                if !traces.is_empty() && node.id.contains("loop") {
                    // If no tag match, return first loop with traces
                    return Some(traces);
                }
            }
        }
        None
    }

    /// Get trace buffer directly from a loop node by node ID.
    pub fn traces_by_node(&self, node_id: &str) -> Option<Vec<Variable>> {
        if let Some(&ni) = self.node_index.get(node_id)
            && let Some(ref buf) = self.nodes[ni].trace_buf
        {
            let traces = buf.borrow().clone();
            if !traces.is_empty() {
                return Some(traces);
            }
        }
        None
    }

    /// Expand tag group names into their member tags.
    /// Non-group tags pass through unchanged.
    pub(crate) fn expand_groups(&self, tags: &[&str]) -> Vec<String> {
        let mut expanded = Vec::new();
        for &tag in tags {
            if let Some(members) = self.tag_groups.get(tag) {
                expanded.extend(members.iter().cloned());
            } else {
                expanded.push(tag.to_string());
            }
        }
        expanded
    }
}
