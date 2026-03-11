use crate::autograd::Variable;
use crate::tensor::Result;

use super::parameter::Parameter;
use super::Module;

/// ReLU activation module.
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.relu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Sigmoid activation module.
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.sigmoid()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Tanh activation module.
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.tanh_act()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// GELU activation (Gaussian Error Linear Unit).
///
/// Approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        GELU
    }
}

impl Module for GELU {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        // sqrt(2/pi) ≈ 0.7978845608
        let x3 = input.mul(input)?.mul(input)?;
        let inner = input.add(&x3.mul_scalar(0.044715)?)?
            .mul_scalar(0.7978845608)?;
        let tanh_inner = inner.tanh_act()?;
        let one_plus = tanh_inner.add_scalar(1.0)?;
        input.mul(&one_plus)?.mul_scalar(0.5)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// SiLU activation (Sigmoid Linear Unit / Swish): `x * sigmoid(x)`
pub struct SiLU;

impl SiLU {
    pub fn new() -> Self {
        SiLU
    }
}

impl Module for SiLU {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let s = input.sigmoid()?;
        input.mul(&s)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}
