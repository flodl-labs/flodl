use crate::autograd::Variable;
use crate::tensor::{Result, Tensor, TensorOptions, DType, Device};

use super::parameter::Parameter;
use super::Module;

/// Lookup table for token embeddings.
///
/// Weight shape: `[num_embeddings, embedding_dim]`.
/// Input: integer indices as an i64 or f32 tensor. Output: embedded vectors.
/// Prefer i64 inputs for vocabularies larger than 16M tokens (f32 loses
/// precision beyond 2^24).
///
/// ```ignore
/// let emb = Embedding::new(1000, 64)?;
/// // Input: [seq_len] of token indices → Output: [seq_len, 64]
/// let indices = Variable::new(Tensor::from_i64(&[0, 5, 42], &[3])?, false);
/// let vectors = emb.forward(&indices)?;
/// assert_eq!(vectors.shape(), vec![3, 64]);
/// ```
pub struct Embedding {
    pub weight: Parameter,
    #[allow(dead_code)]
    num_embeddings: i64,
    embedding_dim: i64,
}

impl Embedding {
    /// Create an embedding table with `num_embeddings` entries of dimension `embedding_dim`.
    pub fn new(num_embeddings: i64, embedding_dim: i64) -> Result<Self> {
        let weight = Variable::new(
            Tensor::randn(
                &[num_embeddings, embedding_dim],
                TensorOptions { dtype: DType::Float32, device: Device::CPU },
            )?,
            true,
        );

        Ok(Embedding {
            weight: Parameter {
                variable: weight,
                name: "weight".into(),
            },
            num_embeddings,
            embedding_dim,
        })
    }
}

impl Module for Embedding {
    fn name(&self) -> &str { "embedding" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Input shape: [*] (any shape of indices)
        // Output shape: [*, embedding_dim]
        let input_shape = input.shape();
        let numel = input.numel();

        // Build i64 index tensor: use native i64 when available, fall back to f32 conversion
        let index_tensor = if input.data().dtype() == DType::Int64 {
            input.data().reshape(&[numel])?
        } else {
            let flat_data = input.data().to_f32_vec()?;
            let indices: Vec<i64> = flat_data.iter().map(|&v| v as i64).collect();
            Tensor::from_i64(&indices, &[numel])?
        };

        // index_select along dim 0
        let selected = self.weight.variable.index_select(0, &index_tensor)?;

        // Reshape to [*input_shape, embedding_dim]
        let mut output_shape = input_shape;
        output_shape.push(self.embedding_dim);
        selected.reshape(&output_shape)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }
}
