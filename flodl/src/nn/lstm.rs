use crate::autograd::Variable;
use crate::tensor::{Device, DType, Result, Tensor, TensorOptions};

use super::lstmcell::LSTMCell;
use super::parameter::Parameter;
use super::Module;

/// Multi-layer LSTM (Long Short-Term Memory) sequence module.
///
/// Wraps multiple [`LSTMCell`] layers and loops over timesteps, matching
/// the PyTorch `nn.LSTM` interface. Each layer feeds its output sequence
/// as input to the next layer.
///
/// ```ignore
/// let lstm = LSTM::new(4, 8, 2)?;       // input=4, hidden=8, 2 layers
/// let x = Variable::new(Tensor::randn(&[10, 1, 4], opts)?, false); // [seq, batch, input]
/// let (output, (h_n, c_n)) = lstm.forward_seq(&x, None)?;
/// // output: [10, 1, 8], h_n: [2, 1, 8], c_n: [2, 1, 8]
/// ```
pub struct LSTM {
    cells: Vec<LSTMCell>,
    hidden_size: i64,
    num_layers: usize,
    batch_first: bool,
}

impl LSTM {
    /// Create a multi-layer LSTM on CPU.
    pub fn new(input_size: i64, hidden_size: i64, num_layers: usize) -> Result<Self> {
        Self::on_device(input_size, hidden_size, num_layers, false, Device::CPU)
    }

    /// Create a multi-layer LSTM on a specific device.
    pub fn on_device(
        input_size: i64,
        hidden_size: i64,
        num_layers: usize,
        batch_first: bool,
        device: Device,
    ) -> Result<Self> {
        assert!(num_layers >= 1, "LSTM requires at least 1 layer");
        let mut cells = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let in_size = if layer == 0 { input_size } else { hidden_size };
            cells.push(LSTMCell::on_device(in_size, hidden_size, device)?);
        }
        Ok(LSTM {
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
    /// - `state_0`: optional initial state `(h_0, c_0)` each `[num_layers, batch, hidden_size]`,
    ///   or None for zeros
    ///
    /// Returns `(output, (h_n, c_n))`:
    /// - `output`: `[seq_len, batch, hidden_size]` (or `[batch, seq_len, hidden_size]` if batch_first)
    /// - `h_n`: `[num_layers, batch, hidden_size]`
    /// - `c_n`: `[num_layers, batch, hidden_size]`
    pub fn forward_seq(
        &self,
        input: &Variable,
        state_0: Option<(&Variable, &Variable)>,
    ) -> Result<(Variable, (Variable, Variable))> {
        // Normalize to [seq_len, batch, features]
        let x = if self.batch_first {
            input.transpose(0, 1)?
        } else {
            input.clone()
        };

        let seq_len = x.shape()[0];
        let batch = x.shape()[1];
        let hs = self.hidden_size;
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: self.cells[0].parameters()[0].variable.device(),
        };

        // Initialize hidden and cell states per layer
        let (mut h, mut c): (Vec<Variable>, Vec<Variable>) = if let Some((h0, c0)) = state_0 {
            let h_vec = (0..self.num_layers)
                .map(|l| h0.select(0, l as i64))
                .collect::<Result<Vec<_>>>()?;
            let c_vec = (0..self.num_layers)
                .map(|l| c0.select(0, l as i64))
                .collect::<Result<Vec<_>>>()?;
            (h_vec, c_vec)
        } else {
            let h_vec = (0..self.num_layers)
                .map(|_| {
                    Ok(Variable::new(
                        Tensor::zeros(&[batch, hs], opts)?,
                        false,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let c_vec = (0..self.num_layers)
                .map(|_| {
                    Ok(Variable::new(
                        Tensor::zeros(&[batch, hs], opts)?,
                        false,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            (h_vec, c_vec)
        };

        // Run through timesteps and layers
        let mut outputs = Vec::with_capacity(seq_len as usize);
        for t in 0..seq_len {
            let mut layer_input = x.select(0, t)?;
            for (l, cell) in self.cells.iter().enumerate() {
                // Pack state for LSTMCell: cat(h, c, dim=1)
                let state = h[l].cat(&c[l], 1)?;
                let new_state = cell.forward_step(&layer_input, Some(&state))?;
                // Unpack: h = state[:, :hs], c = state[:, hs:]
                h[l] = new_state.narrow(1, 0, hs)?;
                c[l] = new_state.narrow(1, hs, hs)?;
                layer_input = h[l].clone();
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

        // Stack final hidden/cell states: [num_layers, batch, hidden_size]
        let h_n = Variable::stack(&h, 0)?;
        let c_n = Variable::stack(&c, 0)?;

        Ok((output, (h_n, c_n)))
    }
}

impl Module for LSTM {
    fn name(&self) -> &str { "lstm" }

    /// Module trait forward: runs the full sequence with zero-initialized state.
    /// Returns only the output sequence (not h_n/c_n). Use [`forward_seq`](LSTM::forward_seq)
    /// for explicit state access.
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let (output, _) = self.forward_seq(input, None)?;
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
    fn test_lstm_shapes() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let (output, (h_n, c_n)) = lstm.forward_seq(&x, None).unwrap();

        assert_eq!(output.shape(), vec![5, 3, 8]); // [seq, batch, hidden]
        assert_eq!(h_n.shape(), vec![2, 3, 8]);    // [layers, batch, hidden]
        assert_eq!(c_n.shape(), vec![2, 3, 8]);    // [layers, batch, hidden]
    }

    #[test]
    fn test_lstm_batch_first() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 2, true, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[3, 5, 4], opts).unwrap(), false);
        let (output, (h_n, c_n)) = lstm.forward_seq(&x, None).unwrap();

        assert_eq!(output.shape(), vec![3, 5, 8]); // [batch, seq, hidden]
        assert_eq!(h_n.shape(), vec![2, 3, 8]);
        assert_eq!(c_n.shape(), vec![2, 3, 8]);
    }

    #[test]
    fn test_lstm_with_initial_state() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let h0 = Variable::new(Tensor::randn(&[2, 3, 8], opts).unwrap(), false);
        let c0 = Variable::new(Tensor::randn(&[2, 3, 8], opts).unwrap(), false);
        let (output, (h_n, c_n)) = lstm.forward_seq(&x, Some((&h0, &c0))).unwrap();

        assert_eq!(output.shape(), vec![5, 3, 8]);
        assert_eq!(h_n.shape(), vec![2, 3, 8]);
        assert_eq!(c_n.shape(), vec![2, 3, 8]);
    }

    #[test]
    fn test_lstm_single_layer() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 1, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let (output, (h_n, c_n)) = lstm.forward_seq(&x, None).unwrap();

        assert_eq!(output.shape(), vec![5, 3, 8]);
        assert_eq!(h_n.shape(), vec![1, 3, 8]);
        assert_eq!(c_n.shape(), vec![1, 3, 8]);
    }

    #[test]
    fn test_lstm_gradient() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), true);
        let (output, _) = lstm.forward_seq(&x, None).unwrap();
        let loss = output.sum().unwrap();
        loss.backward().unwrap();

        for p in lstm.parameters() {
            assert!(p.variable.grad().is_some(), "missing grad for {}", p.name);
        }
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_lstm_module_forward() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let y = lstm.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![5, 3, 8]);
    }

    #[test]
    fn test_lstm_parameters_count() {
        let dev = crate::tensor::test_device();
        let lstm = LSTM::on_device(4, 8, 2, false, dev).unwrap();

        // Layer 0: 4 params (w_ih, w_hh, b_ih, b_hh)
        // Layer 1: 4 params
        assert_eq!(lstm.parameters().len(), 8);
    }

    #[test]
    fn test_lstm_builder_pattern() {
        let dev = crate::tensor::test_device();
        let lstm = LSTM::on_device(4, 8, 1, false, dev).unwrap().batch_first(true);
        let opts = crate::tensor::test_opts();
        let x = Variable::new(Tensor::randn(&[3, 5, 4], opts).unwrap(), false);
        let (output, _) = lstm.forward_seq(&x, None).unwrap();
        assert_eq!(output.shape(), vec![3, 5, 8]); // [batch, seq, hidden]
    }
}
