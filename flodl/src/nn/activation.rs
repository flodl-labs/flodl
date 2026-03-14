use crate::autograd::Variable;
use crate::tensor::Result;

use super::Module;

/// Identity pass-through module. Returns its input unchanged.
///
/// Useful as a tagging entry point in graphs:
/// ```ignore
/// FlowBuilder::from(Identity).tag("image")
/// ```
pub struct Identity;

impl Default for Identity {
    fn default() -> Self {
        Identity
    }
}

impl Identity {
    /// Create an Identity module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Identity {
    fn name(&self) -> &str { "identity" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(input.clone())
    }
}

/// ReLU activation module.
pub struct ReLU;

impl Default for ReLU {
    fn default() -> Self {
        ReLU
    }
}

impl ReLU {
    /// Create a ReLU activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for ReLU {
    fn name(&self) -> &str { "relu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.relu()
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
    /// Create a Sigmoid activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Sigmoid {
    fn name(&self) -> &str { "sigmoid" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.sigmoid()
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
    /// Create a Tanh activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Tanh {
    fn name(&self) -> &str { "tanh" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.tanh_act()
    }
}

/// GELU activation (Gaussian Error Linear Unit).
///
/// Uses the exact form: `0.5 * x * (1 + erf(x / sqrt(2)))`
pub struct GELU;

impl Default for GELU {
    fn default() -> Self {
        GELU
    }
}

impl GELU {
    /// Create a GELU activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for GELU {
    fn name(&self) -> &str { "gelu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.gelu()
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
    /// Create a SiLU activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for SiLU {
    fn name(&self) -> &str { "silu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.silu()
    }
}
