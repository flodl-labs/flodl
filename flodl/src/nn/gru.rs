use crate::autograd::Variable;
use crate::tensor::{Device, DType, Result, Tensor, TensorOptions};

use super::grucell::GRUCell;
use super::parameter::Parameter;
use super::Module;

/// Multi-layer GRU (Gated Recurrent Unit) sequence module.
///
/// Wraps multiple [`GRUCell`] layers and loops over timesteps, matching
/// the PyTorch `nn.GRU` interface. Each layer feeds its output sequence
/// as input to the next layer.
///
/// ```ignore
/// let gru = GRU::new(4, 8, 2)?;       // input=4, hidden=8, 2 layers
/// let x = Variable::new(Tensor::randn(&[10, 1, 4], opts)?, false); // [seq, batch, input]
/// let (output, h_n) = gru.forward_seq(&x, None)?;
/// // output: [10, 1, 8], h_n: [2, 1, 8]
/// ```
pub struct GRU {
    cells: Vec<GRUCell>,
    hidden_size: i64,
    num_layers: usize,
    batch_first: bool,
}

impl GRU {
    /// Create a multi-layer GRU on CPU.
    pub fn new(input_size: i64, hidden_size: i64, num_layers: usize) -> Result<Self> {
        Self::on_device(input_size, hidden_size, num_layers, false, Device::CPU)
    }

    /// Create a multi-layer GRU on a specific device.
    pub fn on_device(
        input_size: i64,
        hidden_size: i64,
        num_layers: usize,
        batch_first: bool,
        device: Device,
    ) -> Result<Self> {
        assert!(num_layers >= 1, "GRU requires at least 1 layer");
        let mut cells = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let in_size = if layer == 0 { input_size } else { hidden_size };
            cells.push(GRUCell::on_device(in_size, hidden_size, device)?);
        }
        Ok(GRU {
            cells,
            hidden_size,
            num_layers,
            batch_first,
        })
    }

    /// Set batch_first mode. When true, input/output are `[batch, seq, features]`.
    pub fn batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
    }

    /// Forward pass over a full sequence.
    ///
    /// - `input`: `[seq_len, batch, input_size]` (or `[batch, seq_len, input_size]` if batch_first)
    /// - `h_0`: optional initial hidden state `[num_layers, batch, hidden_size]`, or None for zeros
    ///
    /// Returns `(output, h_n)`:
    /// - `output`: `[seq_len, batch, hidden_size]` (or `[batch, seq_len, hidden_size]` if batch_first)
    /// - `h_n`: `[num_layers, batch, hidden_size]`
    pub fn forward_seq(
        &self,
        input: &Variable,
        h_0: Option<&Variable>,
    ) -> Result<(Variable, Variable)> {
        // Normalize to [seq_len, batch, features]
        let x = if self.batch_first {
            input.transpose(0, 1)?
        } else {
            input.clone()
        };

        let seq_len = x.shape()[0];
        let batch = x.shape()[1];
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: self.cells[0].parameters()[0].variable.device(),
        };

        // Initialize hidden states per layer
        let mut h: Vec<Variable> = if let Some(h0) = h_0 {
            (0..self.num_layers)
                .map(|l| h0.select(0, l as i64))
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..self.num_layers)
                .map(|_| {
                    Ok(Variable::new(
                        Tensor::zeros(&[batch, self.hidden_size], opts)?,
                        false,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Run through timesteps and layers
        let mut outputs = Vec::with_capacity(seq_len as usize);
        for t in 0..seq_len {
            let mut layer_input = x.select(0, t)?;
            for (l, cell) in self.cells.iter().enumerate() {
                let h_new = cell.forward_step(&layer_input, Some(&h[l]))?;
                h[l] = h_new.clone();
                layer_input = h_new;
            }
            outputs.push(layer_input);
        }

        // Stack outputs: [seq_len, batch, hidden_size]
        let output = Variable::stack(&outputs, 0)?;
        let output = if self.batch_first {
            output.transpose(0, 1)?
        } else {
            output
        };

        // Stack final hidden states: [num_layers, batch, hidden_size]
        let h_n = Variable::stack(&h, 0)?;

        Ok((output, h_n))
    }
}

impl Module for GRU {
    fn name(&self) -> &str { "gru" }

    /// Module trait forward: runs the full sequence with zero-initialized hidden state.
    /// Returns only the output sequence (not h_n). Use [`forward_seq`](GRU::forward_seq)
    /// for explicit hidden state access.
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let (output, _h_n) = self.forward_seq(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gru_shapes() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let gru = GRU::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let (output, h_n) = gru.forward_seq(&x, None).unwrap();

        assert_eq!(output.shape(), vec![5, 3, 8]); // [seq, batch, hidden]
        assert_eq!(h_n.shape(), vec![2, 3, 8]);    // [layers, batch, hidden]
    }

    #[test]
    fn test_gru_batch_first() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let gru = GRU::on_device(4, 8, 2, true, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[3, 5, 4], opts).unwrap(), false);
        let (output, h_n) = gru.forward_seq(&x, None).unwrap();

        assert_eq!(output.shape(), vec![3, 5, 8]); // [batch, seq, hidden]
        assert_eq!(h_n.shape(), vec![2, 3, 8]);    // [layers, batch, hidden]
    }

    #[test]
    fn test_gru_with_initial_hidden() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let gru = GRU::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let h0 = Variable::new(Tensor::randn(&[2, 3, 8], opts).unwrap(), false);
        let (output, h_n) = gru.forward_seq(&x, Some(&h0)).unwrap();

        assert_eq!(output.shape(), vec![5, 3, 8]);
        assert_eq!(h_n.shape(), vec![2, 3, 8]);
    }

    #[test]
    fn test_gru_single_layer() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let gru = GRU::on_device(4, 8, 1, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let (output, h_n) = gru.forward_seq(&x, None).unwrap();

        assert_eq!(output.shape(), vec![5, 3, 8]);
        assert_eq!(h_n.shape(), vec![1, 3, 8]);
    }

    #[test]
    fn test_gru_gradient() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let gru = GRU::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), true);
        let (output, _h_n) = gru.forward_seq(&x, None).unwrap();
        let loss = output.sum().unwrap();
        loss.backward().unwrap();

        for p in gru.parameters() {
            assert!(p.variable.grad().is_some(), "missing grad for {}", p.name);
        }
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_gru_module_forward() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let gru = GRU::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let y = gru.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![5, 3, 8]);
    }

    #[test]
    fn test_gru_parameters_count() {
        let dev = crate::tensor::test_device();
        let gru = GRU::on_device(4, 8, 2, false, dev).unwrap();

        // Layer 0: 4 params (w_ih, w_hh, b_ih, b_hh)
        // Layer 1: 4 params
        assert_eq!(gru.parameters().len(), 8);
    }

    #[test]
    fn test_gru_builder_pattern() {
        let dev = crate::tensor::test_device();
        let gru = GRU::on_device(4, 8, 1, false, dev).unwrap().batch_first(true);
        let opts = crate::tensor::test_opts();
        let x = Variable::new(Tensor::randn(&[3, 5, 4], opts).unwrap(), false);
        let (output, _) = gru.forward_seq(&x, None).unwrap();
        assert_eq!(output.shape(), vec![3, 5, 8]); // [batch, seq, hidden]
    }
}
