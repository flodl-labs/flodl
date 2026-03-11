use crate::autograd::Variable;
use crate::tensor::Result;

use super::parameter::Parameter;
use super::Module;

/// ReLU activation module.
pub struct ReLU;

impl Default for ReLU {
    fn default() -> Self {
        ReLU
    }
}

impl ReLU {
    pub fn new() -> Self {
        Self
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

impl Default for Sigmoid {
    fn default() -> Self {
        Sigmoid
    }
}

impl Sigmoid {
    pub fn new() -> Self {
        Self
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

impl Default for Tanh {
    fn default() -> Self {
        Tanh
    }
}

impl Tanh {
    pub fn new() -> Self {
        Self
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

impl Default for GELU {
    fn default() -> Self {
        GELU
    }
}

impl GELU {
    pub fn new() -> Self {
        Self
    }
}

impl Module for GELU {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.gelu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// SiLU activation (Sigmoid Linear Unit / Swish): `x * sigmoid(x)`
pub struct SiLU;

impl Default for SiLU {
    fn default() -> Self {
        SiLU
    }
}

impl SiLU {
    pub fn new() -> Self {
        Self
    }
}

impl Module for SiLU {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.silu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}
