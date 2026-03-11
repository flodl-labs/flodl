use crate::autograd::Variable;
use crate::tensor::{Result, Tensor, TensorOptions, DType, Device};

use super::parameter::Parameter;
use super::Module;

/// Lookup table for token embeddings.
///
/// Weight shape: `[num_embeddings, embedding_dim]`.
/// Input: integer indices (as f32 tensor). Output: embedded vectors.
pub struct Embedding {
    pub weight: Parameter,
    #[allow(dead_code)]
    num_embeddings: i64,
    embedding_dim: i64,
}

impl Embedding {
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
    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Input shape: [*] (any shape of indices)
        // Output shape: [*, embedding_dim]
        let input_shape = input.shape();

        // Flatten input to 1D for index_select
        let numel = input.numel();
        let flat_data = input.data().to_f32_vec()?;
        let indices: Vec<i64> = flat_data.iter().map(|&v| v as i64).collect();

        // Create index tensor (i64 stored as f32 for our from_f32, but index_select needs i64)
        // We need to create an i64 tensor. Let me use from_f32 and cast...
        // Actually, the C++ index_select expects Long tensor. Let me create it via from_blob.
        let index_tensor = Tensor::from_i64(&indices, &[numel])?;

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
