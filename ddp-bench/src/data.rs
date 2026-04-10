//! Synthetic dataset generation for reproducible benchmarks.
//!
//! Uses bulk tensor operations for fast pool generation.

use std::sync::Arc;

use flodl::data::BatchDataSet;
use flodl::tensor::{Device, Result, Tensor, TensorOptions};

/// A pre-generated synthetic dataset stored as bulk tensors.
///
/// Each tensor group is `[total_samples, ...]`. Batching uses `index_select`
/// on the pre-stacked pool, avoiding per-sample stack overhead.
pub struct SyntheticDataSet {
    /// tensors[group_idx] = [total_samples, per-sample dims...]
    tensors: Vec<Tensor>,
    len: usize,
}

impl SyntheticDataSet {
    /// Generate a regression dataset: input [input_dim], target [output_dim].
    ///
    /// Uses independent random tensors. Training still produces gradients
    /// and the model learns a mapping, which is all DDP benchmarks need.
    pub fn regression(
        seed: u64,
        total_samples: usize,
        input_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = total_samples as i64;
        let opts = TensorOptions::default();

        let inputs = Tensor::randn(&[n, input_dim], opts)?;
        let targets = Tensor::randn(&[n, output_dim], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            len: total_samples,
        }))
    }

    /// Generate a classification dataset: input [dims...], target [] (class index).
    pub fn classification(
        seed: u64,
        total_samples: usize,
        input_shape: &[i64],
        num_classes: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = total_samples as i64;
        let opts = TensorOptions::default();

        let mut shape = vec![n];
        shape.extend_from_slice(input_shape);
        let inputs = Tensor::randn(&shape, opts)?;
        let targets = Tensor::randint(0, num_classes, &[n], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            len: total_samples,
        }))
    }

    /// Generate a reconstruction dataset: input [dims...], target = input.
    pub fn reconstruction(
        seed: u64,
        total_samples: usize,
        shape: &[i64],
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = total_samples as i64;
        let opts = TensorOptions::default();

        let mut full_shape = vec![n];
        full_shape.extend_from_slice(shape);
        let inputs = Tensor::randn(&full_shape, opts)?;
        let targets = inputs.clone();

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            len: total_samples,
        }))
    }

    /// Generate a sequence dataset: input [seq_len, input_dim], target [output_dim].
    pub fn sequence(
        seed: u64,
        total_samples: usize,
        seq_len: i64,
        input_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = total_samples as i64;
        let opts = TensorOptions::default();

        let inputs = Tensor::randn(&[n, seq_len, input_dim], opts)?;
        let targets = Tensor::randn(&[n, output_dim], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            len: total_samples,
        }))
    }

    /// Generate a token sequence dataset: input [seq_len] (i64 tokens), target [seq_len].
    pub fn token_sequence(
        seed: u64,
        total_samples: usize,
        seq_len: i64,
        vocab_size: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = total_samples as i64;
        let opts = TensorOptions::default();

        let inputs = Tensor::randint(0, vocab_size, &[n, seq_len], opts)?;
        let targets = Tensor::randint(0, vocab_size, &[n, seq_len], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            len: total_samples,
        }))
    }
}

impl BatchDataSet for SyntheticDataSet {
    fn len(&self) -> usize {
        self.len
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        // Build index tensor (wrapping via modulo)
        let idx: Vec<i64> = indices
            .iter()
            .map(|&i| (i % self.len) as i64)
            .collect();
        let idx_tensor = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;

        let mut result = Vec::with_capacity(self.tensors.len());
        for bulk in &self.tensors {
            result.push(bulk.index_select(0, &idx_tensor)?);
        }
        Ok(result)
    }
}
