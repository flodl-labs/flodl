use std::ffi::c_void;
use std::io::{Read, Write};
use std::ptr;

use flodl_sys as ffi;

use crate::autograd::Variable;
use crate::tensor::{DType, Device, Result, Tensor};

use super::checkpoint::{read_f64_le, read_i64_le, write_f64_le, write_i64_le};
use super::optim::Stateful;
use super::parameter::Parameter;

/// RAII guard that enables automatic mixed precision.
///
/// While active, eligible operations (matmul, conv, linear) dispatch to the
/// given reduced-precision dtype. Numerically sensitive operations (losses,
/// norms, softmax) stay in fp32 automatically.
///
/// Requires Tensor Core hardware (Volta/Turing/Ampere/Ada/Blackwell) for
/// actual speedup. On older GPUs or CPU, ops still run but without perf gain.
///
/// ```ignore
/// let _amp = AutocastGuard::new(DType::Float16);
/// let output = model.forward(&input)?;  // matmul runs in fp16
/// let loss = mse_loss(&output, &target)?;  // stays fp32
/// drop(_amp);
/// ```
pub struct AutocastGuard {
    guard: *mut c_void,
}

impl AutocastGuard {
    /// Enable CUDA autocast with the given dtype (typically `Float16` or `BFloat16`).
    pub fn new(dtype: DType) -> Self {
        let guard = unsafe { ffi::flodl_autocast_guard_new(ffi::FLODL_CUDA, dtype as i32) };
        AutocastGuard { guard }
    }

    /// Enable autocast for a specific device type with the given dtype.
    pub fn for_device(device_type: i32, dtype: DType) -> Self {
        let guard = unsafe { ffi::flodl_autocast_guard_new(device_type, dtype as i32) };
        AutocastGuard { guard }
    }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        if !self.guard.is_null() {
            unsafe { ffi::flodl_autocast_guard_delete(self.guard) };
            self.guard = ptr::null_mut();
        }
    }
}

/// Run a closure with CUDA autocast enabled.
///
/// Equivalent to creating an [`AutocastGuard`] for the duration of `f`.
///
/// ```ignore
/// let loss = autocast(DType::Float16, || {
///     let output = model.forward(&input)?;
///     mse_loss(&output, &target)
/// })?;
/// ```
pub fn autocast<F, R>(dtype: DType, f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = AutocastGuard::new(dtype);
    f()
}

/// Returns true if CUDA autocast is currently enabled.
///
/// CUDA shorthand for [`is_autocast_enabled_for`]; use that to query a
/// different device type (mirrors [`AutocastGuard::for_device`]).
pub fn is_autocast_enabled() -> bool {
    is_autocast_enabled_for(ffi::FLODL_CUDA)
}

/// Returns true if autocast is currently enabled for the given device type
/// (`ffi::FLODL_CUDA` / `ffi::FLODL_CPU`). Mirrors
/// [`AutocastGuard::for_device`], which sets it per device type.
pub fn is_autocast_enabled_for(device_type: i32) -> bool {
    unsafe { ffi::flodl_is_autocast_enabled(device_type) != 0 }
}

/// The numeric precision a training step computes in.
///
/// This is *mixed* precision: parameters, gradients, checkpoints and
/// averaging stay `f32` on every variant, and only the forward pass is
/// allowed a narrower type (libtorch's autocast, which keeps losses,
/// norms and softmax in `f32` on its own). Storing the model itself in
/// another type is a different axis and not expressed here.
///
/// A reduced precision pays on matmul-heavy models (transformers) on
/// hardware that computes it natively. A small convolutional model can
/// get slower, and a card that only emulates the type gets much slower
/// (see [`Precision::is_native_on`]), so `F32` is the default.
///
/// ```ignore
/// let loss = Precision::Bf16Mixed.run(device, || {
///     let out = model.forward(&input)?;
///     cross_entropy_loss(&out, &target)
/// })?;
/// loss.backward()?; // outside, as with autocast
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Full `f32` compute. The default.
    #[default]
    F32,
    /// `bf16` forward under autocast, `f32` parameters. Needs no
    /// gradient scaler: `bf16` keeps `f32`'s exponent range.
    Bf16Mixed,
    /// `f16` forward under autocast, `f32` parameters, with the loss
    /// scaled before `backward` and the gradients unscaled before the
    /// optimizer step ([`GradScaler`]): `f16`'s narrow exponent range
    /// would otherwise round small gradients to zero. The trainer does
    /// the scaling; a hand-written loop drives a [`GradScaler`] itself.
    /// Often the fastest choice on convolutional models, where `bf16`
    /// can lose to `f32`.
    F16Mixed,
}

impl Precision {
    /// Every variant, in declaration order.
    pub const ALL: &'static [Precision] =
        &[Precision::F32, Precision::Bf16Mixed, Precision::F16Mixed];

    /// The name used on command lines and in run metadata (`f32`,
    /// `bf16-mixed`, `f16-mixed`). Round-trips through [`FromStr`](std::str::FromStr).
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::Bf16Mixed => "bf16-mixed",
            Precision::F16Mixed => "f16-mixed",
        }
    }

    /// The autocast dtype of the forward pass, `None` for full precision.
    pub fn autocast_dtype(self) -> Option<DType> {
        match self {
            Precision::F32 => None,
            Precision::Bf16Mixed => Some(DType::BFloat16),
            Precision::F16Mixed => Some(DType::Float16),
        }
    }

    /// Whether training at this precision needs the loss scaled and the
    /// gradients unscaled around `backward` (a [`GradScaler`]). True for
    /// `f16` only: its smallest normal value is about 6e-5, and gradients
    /// below that vanish unless the loss is scaled up first.
    pub fn needs_grad_scaler(self) -> bool {
        matches!(self, Precision::F16Mixed)
    }

    /// Whether `device` computes this precision natively, as opposed to
    /// emulating it.
    ///
    /// `bf16` is native on NVIDIA from compute capability 8.0 (Ampere)
    /// and emulated below it, and native on every AMD card: the same
    /// answers as PyTorch's `torch.cuda.is_bf16_supported(
    /// including_emulation=False)`. Emulation is correct but slow: a GTX 1060 trains a small
    /// transformer about 1.3x slower in `bf16` than in `f32`, and a small
    /// convolutional model several times slower. A device that cannot
    /// answer (an AMD build, where compute capability does not exist, or
    /// the CPU) counts as native.
    pub fn is_native_on(self, device: Device) -> bool {
        match self {
            Precision::F32 => true,
            Precision::Bf16Mixed => match device {
                Device::CUDA(idx) => crate::tensor::cuda_compute_capability(idx as i32)
                    .is_none_or(|(major, _)| major >= 8),
                // No compute-capability query. MPS counts as native: Apple
                // GPUs run bf16, and a device that cannot answer is not
                // treated as emulating.
                Device::CPU | Device::MPS => true,
            },
            // Every CUDA and ROCm card computes `f16`. Cards without
            // tensor cores do it slowly, which is a speed matter the
            // caller measures, not an emulation this can detect.
            Precision::F16Mixed => true,
        }
    }

    /// The precision `device` should run when the caller prefers `f32`
    /// over an emulated type: `self` where it is native, `F32` otherwise.
    pub fn native_or_f32(self, device: Device) -> Precision {
        if self.is_native_on(device) {
            self
        } else {
            Precision::F32
        }
    }

    /// Run a forward pass (forward plus loss) at this precision on
    /// `device`. Call `backward` outside the closure.
    pub fn run<F, R>(self, device: Device, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        match self.autocast_dtype() {
            Some(dtype) => {
                let _guard = AutocastGuard::for_device(device.to_ffi().0, dtype);
                f()
            }
            None => f(),
        }
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Precision {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, String> {
        Precision::ALL
            .iter()
            .copied()
            .find(|p| p.as_str() == s)
            .ok_or_else(|| {
                let names: Vec<&str> = Precision::ALL.iter().map(|p| p.as_str()).collect();
                format!(
                    "unknown precision `{s}` (expected one of: {})",
                    names.join(", ")
                )
            })
    }
}

/// Cast all parameters to a different dtype. No-op for parameters already
/// at the target dtype.
///
/// All-or-nothing: every conversion is computed first, so if any fails the
/// error is returned with NO parameter mutated. Previously a per-parameter
/// failure was silently swallowed, leaving a half-cast, mixed-dtype model
/// that failed cryptically much later.
pub fn cast_parameters(params: &[Parameter], dtype: DType) -> Result<()> {
    let mut pending: Vec<(&Parameter, Tensor)> = Vec::new();
    for p in params {
        if p.variable.data().dtype() != dtype {
            pending.push((p, p.variable.data().to_dtype(dtype)?));
        }
    }
    for (p, t) in pending {
        p.variable.set_data(t);
    }
    Ok(())
}

/// GradScaler for mixed precision training.
///
/// Scales loss before backward to prevent gradient underflow in float16,
/// then unscales gradients before optimizer step. Dynamically adjusts
/// scale factor based on whether inf/nan gradients are detected.
///
/// ```ignore
/// let mut scaler = GradScaler::new();
/// let scaled_loss = scaler.scale(&loss)?;
/// scaled_loss.backward()?;
/// let stepped = scaler.step(&params, &mut || optim.step())?;
/// scaler.update();
/// ```
pub struct GradScaler {
    scale: f64,
    growth: f64,
    backoff: f64,
    interval: i64,
    steps_since_growth: i64,
    found_inf: bool,
    /// Whether this step's gradients were already unscaled, so that
    /// [`Self::step`] after an explicit [`Self::unscale_`] does not divide
    /// them twice. Cleared by [`Self::update`].
    unscaled: bool,
    /// Lower bound on `scale`. Loss scaling only ever scales *up* (to lift
    /// small fp16 gradients out of underflow), so the scale should never
    /// need to drop below 1.0. Without a floor, sustained inf/NaN gradients
    /// halve the scale every step until it underflows to 0.0, after which
    /// `scale(loss)` is always 0, gradients are always "finite" zeros, and
    /// training is silently dead. Flooring keeps the scale usable; if inf
    /// persists at the floor, the divergence is real (not a scaling issue).
    min_scale: f64,
}

impl Default for GradScaler {
    fn default() -> Self {
        GradScaler {
            scale: 65536.0,
            growth: 2.0,
            backoff: 0.5,
            interval: 2000,
            steps_since_growth: 0,
            found_inf: false,
            unscaled: false,
            min_scale: 1.0,
        }
    }
}

impl GradScaler {
    /// Create a new GradScaler with default settings.
    ///
    /// Initial scale: 2^16 = 65536, growth: 2.0, backoff: 0.5, interval: 2000.
    pub fn new() -> Self {
        Self::default()
    }

    /// Scale the loss before backward. Returns loss * scale.
    pub fn scale(&self, loss: &Variable) -> Result<Variable> {
        loss.mul_scalar(self.scale)
    }

    /// Current scale factor.
    pub fn scale_factor(&self) -> f64 {
        self.scale
    }

    /// Divide the gradients of `params` by the current scale, in place,
    /// and record whether any of them is NaN or infinite. Returns `true`
    /// when they are all finite.
    ///
    /// Call it after `backward` on the scaled loss when something has to
    /// see the true gradients before the optimizer does, gradient clipping
    /// above all: a clip applied to scaled gradients cuts them down to a
    /// norm the unscale then divides by the scale again. [`Self::step`]
    /// unscales by itself when this was not called, and does not unscale a
    /// second time when it was.
    ///
    /// The whole list costs one kernel and one host read.
    pub fn unscale_(&mut self, params: &[Parameter]) -> Result<bool> {
        if self.unscaled {
            return Ok(!self.found_inf);
        }
        let with_grad: Vec<Tensor> = params
            .iter()
            .filter(|p| p.variable.grad().is_some())
            .map(|p| p.variable.data())
            .collect();
        let found_inf = Tensor::amp_unscale_grads_(&with_grad, 1.0 / self.scale)?;
        self.found_inf |= found_inf;
        self.unscaled = true;
        Ok(!self.found_inf)
    }

    /// Unscale gradients, check for inf/nan, and step the optimizer.
    ///
    /// Returns true if the step was taken (all gradients finite).
    /// Returns false if inf/nan detected (optimizer step skipped).
    pub fn step(
        &mut self,
        params: &[Parameter],
        step_fn: &mut dyn FnMut() -> Result<()>,
    ) -> Result<bool> {
        if !self.unscale_(params)? {
            return Ok(false);
        }
        step_fn()?;
        Ok(true)
    }

    /// Update the scale factor after each step.
    ///
    /// Call this after every `step()` call, regardless of whether it succeeded.
    pub fn update(&mut self) {
        if self.found_inf {
            // Floor at `min_scale` so sustained inf can't drive the scale to
            // 0.0 (which would silently zero all gradients and kill training).
            self.scale = (self.scale * self.backoff).max(self.min_scale);
            self.steps_since_growth = 0;
        } else {
            self.steps_since_growth += 1;
            if self.steps_since_growth >= self.interval {
                self.scale *= self.growth;
                self.steps_since_growth = 0;
            }
        }
        self.found_inf = false;
        self.unscaled = false;
    }
}

impl Stateful for GradScaler {
    fn state_kind(&self) -> crate::nn::StateKind {
        crate::nn::StateKind::GradScaler
    }

    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_f64_le(w, self.scale)?;
        write_i64_le(w, self.steps_since_growth)?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.scale = read_f64_le(r)?;
        self.steps_since_growth = read_i64_le(r)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_autocast_guard_lifecycle() {
        assert!(!is_autocast_enabled());
        {
            let _guard = AutocastGuard::new(DType::Float16);
            assert!(is_autocast_enabled());
        }
        // Guard dropped, autocast should be disabled
        assert!(!is_autocast_enabled());
    }

    #[test]
    // The closure is the subject, not an accident: this test covers the
    // `F: FnOnce() -> R` entry point, where `test_autocast_guard_lifecycle`
    // above covers the RAII one. Clippy's suggestion (pass the function
    // itself) compiles and asserts the same thing while no longer passing a
    // closure, which leaves the case this test is named for uncovered.
    #[allow(clippy::redundant_closure)]
    fn test_autocast_closure() {
        assert!(!is_autocast_enabled());
        let was_enabled = autocast(DType::Float16, || is_autocast_enabled());
        assert!(was_enabled);
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_cast_parameters() {
        let t = Tensor::ones(&[4], test_opts()).unwrap();
        let p = Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        };
        assert_eq!(p.variable.data().dtype(), DType::Float32);
        cast_parameters(std::slice::from_ref(&p), DType::Float64).unwrap();
        assert_eq!(p.variable.data().dtype(), DType::Float64);
    }

    #[test]
    fn a_cast_parameter_still_trains() {
        // The cast runs with gradients enabled, so its result is a
        // tracked non-leaf. Installed as is, the parameter would never
        // receive a gradient and the model would be silently frozen.
        let dev = crate::tensor::test_device();
        let p = crate::nn::parameter::Parameter::new(
            Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(),
            "p",
        );
        cast_parameters(std::slice::from_ref(&p), DType::Float64).unwrap();
        assert!(p.variable.data().is_leaf());
        let loss = p.variable.mul(&p.variable).unwrap().sum().unwrap();
        loss.backward().unwrap();
        let grad = p
            .variable
            .grad()
            .expect("a cast parameter must receive a gradient");
        assert_eq!(grad.to_f64_vec().unwrap(), vec![2.0, 4.0]);
    }

    #[test]
    fn test_cast_parameters_noop_same_dtype() {
        let t = Tensor::ones(&[4], test_opts()).unwrap();
        let p = Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        };
        cast_parameters(std::slice::from_ref(&p), DType::Float32).unwrap();
        assert_eq!(p.variable.data().dtype(), DType::Float32);
    }

    #[test]
    fn test_grad_scaler_defaults() {
        let scaler = GradScaler::new();
        assert_eq!(scaler.scale_factor(), 65536.0);
    }

    #[test]
    fn test_grad_scaler_scale() {
        let scaler = GradScaler::new();
        let loss = Variable::new(Tensor::from_f32(&[1.0], &[1], test_device()).unwrap(), true);
        let scaled = scaler.scale(&loss).unwrap();
        assert!((scaled.item().unwrap() - 65536.0).abs() < 1.0);
    }

    #[test]
    fn test_grad_scaler_step_finite() {
        let mut scaler = GradScaler::new();
        let t = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let p = Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        };
        // Set a finite gradient
        let grad = Tensor::from_f32(&[0.1, 0.2], &[2], test_device()).unwrap();
        p.variable.set_grad(grad);

        let mut stepped = false;
        let ok = scaler
            .step(std::slice::from_ref(&p), &mut || {
                stepped = true;
                Ok(())
            })
            .unwrap();
        assert!(ok);
        assert!(stepped);
    }

    fn param(values: &[f32]) -> Parameter {
        let t = Tensor::from_f32(values, &[values.len() as i64], test_device()).unwrap();
        Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        }
    }

    #[test]
    fn the_scaler_returns_the_true_gradients_of_a_scaled_loss() {
        // loss = sum(p * p), so d loss / d p = 2p. The scaled loss gives
        // gradients 65536 times that; unscaling brings them back.
        let mut scaler = GradScaler::new();
        let (a, b, untouched) = (param(&[1.0, 2.0]), param(&[-3.0]), param(&[5.0]));
        let loss = a
            .variable
            .mul(&a.variable)
            .unwrap()
            .sum()
            .unwrap()
            .add(&b.variable.mul(&b.variable).unwrap().sum().unwrap())
            .unwrap();
        scaler.scale(&loss).unwrap().backward().unwrap();
        let scaled = a.variable.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(scaled, vec![2.0 * 65536.0, 4.0 * 65536.0]);

        let params = [a.clone(), b.clone(), untouched.clone()];
        assert!(scaler.unscale_(&params).unwrap(), "finite gradients");
        assert_eq!(
            a.variable.grad().unwrap().to_f32_vec().unwrap(),
            vec![2.0, 4.0]
        );
        assert_eq!(b.variable.grad().unwrap().to_f32_vec().unwrap(), vec![-6.0]);
        // A parameter the loss never reached has no gradient, before or after.
        assert!(untouched.variable.grad().is_none());
        scaler.update();
        assert_eq!(scaler.scale_factor(), 65536.0);
    }

    #[test]
    fn one_bad_gradient_among_many_skips_the_step_and_backs_the_scale_off() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut scaler = GradScaler::new();
            let params = [param(&[1.0, 2.0]), param(&[3.0]), param(&[4.0])];
            let dev = test_device();
            params[0]
                .variable
                .set_grad(Tensor::from_f32(&[0.5, 0.5], &[2], dev).unwrap());
            params[1]
                .variable
                .set_grad(Tensor::from_f32(&[bad], &[1], dev).unwrap());
            params[2]
                .variable
                .set_grad(Tensor::from_f32(&[0.5], &[1], dev).unwrap());

            let mut stepped = false;
            let ok = scaler
                .step(&params, &mut || {
                    stepped = true;
                    Ok(())
                })
                .unwrap();
            assert!(!ok && !stepped, "{bad}: the step must be skipped");
            scaler.update();
            assert_eq!(scaler.scale_factor(), 32768.0, "{bad}: one backoff");
            // The flag is per step: a clean step afterwards goes through.
            for p in &params {
                p.variable
                    .set_grad(Tensor::from_f32(&[0.5], &[1], dev).unwrap());
            }
            assert!(scaler.step(&params, &mut || Ok(())).unwrap());
        }
    }

    #[test]
    fn unscaling_before_a_clip_is_not_repeated_by_the_step() {
        let mut scaler = GradScaler::new();
        let p = param(&[1.0, 2.0]);
        let loss = p.variable.mul(&p.variable).unwrap().sum().unwrap();
        scaler.scale(&loss).unwrap().backward().unwrap();

        let params = [p.clone()];
        assert!(scaler.unscale_(&params).unwrap());
        // Where a clip would go: the gradients are the true ones.
        assert_eq!(
            p.variable.grad().unwrap().to_f32_vec().unwrap(),
            vec![2.0, 4.0]
        );
        let mut seen = Vec::new();
        assert!(
            scaler
                .step(&params, &mut || {
                    seen = p.variable.grad().unwrap().to_f32_vec().unwrap();
                    Ok(())
                })
                .unwrap()
        );
        assert_eq!(seen, vec![2.0, 4.0], "the step must not unscale again");
        scaler.update();

        // The next step unscales afresh.
        p.variable
            .set_grad(Tensor::from_f32(&[65536.0, 65536.0], &[2], test_device()).unwrap());
        assert!(scaler.step(&params, &mut || Ok(())).unwrap());
        assert_eq!(
            p.variable.grad().unwrap().to_f32_vec().unwrap(),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn precision_names_round_trip_and_only_f16_needs_a_scaler() {
        for &p in Precision::ALL {
            assert_eq!(p.as_str().parse::<Precision>().unwrap(), p);
        }
        assert_eq!(Precision::F16Mixed.as_str(), "f16-mixed");
        assert_eq!(Precision::F16Mixed.autocast_dtype(), Some(DType::Float16));
        assert!(Precision::F16Mixed.needs_grad_scaler());
        assert!(!Precision::Bf16Mixed.needs_grad_scaler());
        assert!(!Precision::F32.needs_grad_scaler());
    }

    #[test]
    fn test_grad_scaler_scale_floors_and_never_dies() {
        // Regression: sustained inf must NOT drive the scale to 0.0 (which
        // would silently zero every gradient and kill training). It floors
        // at min_scale (1.0) instead.
        let mut scaler = GradScaler::new();
        for _ in 0..40 {
            scaler.found_inf = true; // simulate an inf-detected step
            scaler.update();
            assert!(
                scaler.scale_factor() > 0.0,
                "scale must never reach 0.0, got {}",
                scaler.scale_factor()
            );
        }
        assert!(
            (scaler.scale_factor() - 1.0).abs() < 1e-9,
            "sustained inf should settle at the 1.0 floor, got {}",
            scaler.scale_factor()
        );
    }

    #[test]
    fn test_grad_scaler_step_inf() {
        let mut scaler = GradScaler::new();
        let t = Tensor::from_f32(&[1.0], &[1], test_device()).unwrap();
        let p = Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        };
        // Set an infinite gradient
        let grad = Tensor::from_f32(&[f32::INFINITY], &[1], test_device()).unwrap();
        p.variable.set_grad(grad);

        let mut stepped = false;
        let ok = scaler
            .step(&[p], &mut || {
                stepped = true;
                Ok(())
            })
            .unwrap();
        assert!(!ok, "step should be skipped on inf");
        assert!(!stepped, "optimizer should not have stepped");
    }

    #[test]
    fn test_grad_scaler_update_growth() {
        let mut scaler = GradScaler {
            scale: 100.0,
            growth: 2.0,
            backoff: 0.5,
            interval: 3,
            steps_since_growth: 2,
            found_inf: false,
            unscaled: false,
            min_scale: 1.0,
        };
        scaler.update(); // steps_since_growth becomes 3 >= interval
        assert_eq!(scaler.scale_factor(), 200.0);
    }

    #[test]
    fn test_grad_scaler_update_backoff() {
        let mut scaler = GradScaler::new();
        let initial = scaler.scale_factor();
        scaler.found_inf = true;
        scaler.update();
        assert_eq!(scaler.scale_factor(), initial * 0.5);
    }

    #[test]
    fn test_grad_scaler_state_roundtrip() {
        let mut scaler = GradScaler::new();
        // Modify state
        scaler.found_inf = true;
        scaler.update(); // backoff
        scaler.update(); // one good step

        let mut buf = Vec::new();
        scaler.save_state(&mut buf).unwrap();

        let mut loaded = GradScaler::new();
        loaded.load_state(&mut &buf[..]).unwrap();
        assert_eq!(loaded.scale_factor(), scaler.scale_factor());
    }
}
