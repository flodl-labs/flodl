//! NCCL collective operations for multi-GPU communication.
//!
//! Provides AllReduce, Broadcast, and other collective ops across
//! multiple CUDA devices within a single process. Built directly
//! on NCCL for minimal overhead.
//!
//! CUDA only. Requires 2+ GPUs at runtime.
//!
//! # Usage
//!
//! ```ignore
//! let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)])?;
//!
//! // Broadcast initial parameters from device 0 to device 1
//! comms.broadcast(&[&tensor_dev0, &tensor_dev1], 0)?;
//!
//! // AllReduce gradients (sum across devices)
//! comms.all_reduce(&[&grad_dev0, &grad_dev1], ReduceOp::Sum)?;
//! ```

use std::ffi::c_void;
use std::ptr;

use flodl_sys::{self as ffi, FlodlTensor};

use crate::tensor::{check_err, Device, Result, Tensor, TensorError};
use super::cuda_stream::CudaStream;

/// NCCL reduction operation.
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum ReduceOp {
    /// Element-wise sum across devices.
    Sum = 0,
    /// Element-wise product across devices.
    Prod = 1,
    /// Element-wise maximum across devices.
    Max = 2,
    /// Element-wise minimum across devices.
    Min = 3,
    /// Element-wise average across devices.
    Avg = 4,
}

/// NCCL communicator group for multi-GPU collective operations.
///
/// Holds one communicator per device. All collective ops operate
/// across all devices in the group simultaneously.
///
/// RAII: communicators are destroyed on drop.
pub struct NcclComms {
    handle: *mut c_void,
    devices: Vec<Device>,
}

// NcclComms can be sent between threads. The underlying ncclComm_t handles
// are used from the thread that calls the collective ops (with GroupStart/End).
unsafe impl Send for NcclComms {}

impl NcclComms {
    /// Initialize NCCL communicators for the given CUDA devices.
    ///
    /// All devices must be distinct CUDA devices. Returns error on CPU
    /// builds or if NCCL initialization fails.
    pub fn new(devices: &[Device]) -> Result<Self> {
        if devices.len() < 2 {
            return Err(TensorError::new(
                "NcclComms requires at least 2 devices",
            ));
        }
        let mut devlist: Vec<i32> = Vec::with_capacity(devices.len());
        for &dev in devices {
            match dev {
                Device::CUDA(idx) => devlist.push(idx as i32),
                Device::CPU => {
                    return Err(TensorError::new(
                        "NcclComms requires CUDA devices, got CPU",
                    ))
                }
            }
        }

        let mut handle: *mut c_void = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_nccl_init(
                devlist.len() as i32,
                devlist.as_ptr(),
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(NcclComms {
            handle,
            devices: devices.to_vec(),
        })
    }

    /// In-place AllReduce across all devices using default streams.
    ///
    /// Each tensor must reside on its corresponding device and all tensors
    /// must have the same shape and dtype. After completion, every tensor
    /// holds the reduced result.
    pub fn all_reduce(&self, tensors: &[&Tensor], op: ReduceOp) -> Result<()> {
        self.validate_tensors(tensors, "all_reduce")?;
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_nccl_all_reduce(
                self.handle,
                handles.as_mut_ptr(),
                ptr::null_mut(), // default streams
                op as i32,
            )
        };
        check_err(err)
    }

    /// In-place AllReduce on explicit streams (for overlapping with compute).
    ///
    /// Each stream must belong to the corresponding device.
    pub fn all_reduce_on_streams(
        &self,
        tensors: &[&Tensor],
        op: ReduceOp,
        streams: &[&CudaStream],
    ) -> Result<()> {
        self.validate_tensors(tensors, "all_reduce_on_streams")?;
        if streams.len() != self.devices.len() {
            return Err(TensorError::new(&format!(
                "all_reduce_on_streams: expected {} streams, got {}",
                self.devices.len(), streams.len()
            )));
        }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut stream_ptrs: Vec<*mut c_void> = streams.iter().map(|s| s.as_ptr()).collect();
        let err = unsafe {
            ffi::flodl_nccl_all_reduce(
                self.handle,
                handles.as_mut_ptr(),
                stream_ptrs.as_mut_ptr(),
                op as i32,
            )
        };
        check_err(err)
    }

    /// Broadcast tensor from `root` device to all others (in-place).
    ///
    /// After completion, all tensors hold the value that was on `tensors[root]`.
    pub fn broadcast(&self, tensors: &[&Tensor], root: usize) -> Result<()> {
        self.validate_tensors(tensors, "broadcast")?;
        if root >= self.devices.len() {
            return Err(TensorError::new(&format!(
                "broadcast: root {} out of range (have {} devices)",
                root, self.devices.len()
            )));
        }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_nccl_broadcast(
                self.handle,
                handles.as_mut_ptr(),
                ptr::null_mut(), // default streams
                root as i32,
            )
        };
        check_err(err)
    }

    /// Broadcast on explicit streams.
    pub fn broadcast_on_streams(
        &self,
        tensors: &[&Tensor],
        root: usize,
        streams: &[&CudaStream],
    ) -> Result<()> {
        self.validate_tensors(tensors, "broadcast_on_streams")?;
        if root >= self.devices.len() {
            return Err(TensorError::new(&format!(
                "broadcast_on_streams: root {} out of range", root
            )));
        }
        if streams.len() != self.devices.len() {
            return Err(TensorError::new(&format!(
                "broadcast_on_streams: expected {} streams, got {}",
                self.devices.len(), streams.len()
            )));
        }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut stream_ptrs: Vec<*mut c_void> = streams.iter().map(|s| s.as_ptr()).collect();
        let err = unsafe {
            ffi::flodl_nccl_broadcast(
                self.handle,
                handles.as_mut_ptr(),
                stream_ptrs.as_mut_ptr(),
                root as i32,
            )
        };
        check_err(err)
    }

    /// Number of devices in this communicator group.
    pub fn size(&self) -> usize {
        self.devices.len()
    }

    /// Devices in this communicator group.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    fn validate_tensors(&self, tensors: &[&Tensor], op: &str) -> Result<()> {
        if tensors.len() != self.devices.len() {
            return Err(TensorError::new(&format!(
                "{}: expected {} tensors (one per device), got {}",
                op, self.devices.len(), tensors.len()
            )));
        }
        Ok(())
    }
}

impl Drop for NcclComms {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::flodl_nccl_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{test_device, cuda_device_count, cuda_synchronize, TensorOptions, DType};

    fn require_multi_gpu() -> bool {
        if !test_device().is_cuda() || cuda_device_count() < 2 {
            return false;
        }
        // Verify all devices can run compute kernels (e.g., GTX 1060
        // sm_61 is unsupported by libtorch cu128 builds).
        for i in 0..2 {
            let opts = TensorOptions { dtype: DType::Float32, device: Device::CUDA(i) };
            if Tensor::zeros(&[1], opts).is_err() {
                eprintln!("Device CUDA({i}) cannot run compute kernels, skipping multi-GPU test");
                return false;
            }
        }
        true
    }

    #[test]
    fn test_nccl_requires_two_devices() {
        let result = NcclComms::new(&[Device::CUDA(0)]);
        assert!(result.is_err(), "NcclComms should require 2+ devices");
    }

    #[test]
    fn test_nccl_rejects_cpu() {
        let result = NcclComms::new(&[Device::CPU, Device::CPU]);
        assert!(result.is_err(), "NcclComms should reject CPU devices");
    }

    #[test]
    #[ignore = "NCCL needs 2+ GPUs; run with: make cuda-test-nccl"]
    fn test_nccl_init_destroy() {
        if !require_multi_gpu() { return; }
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();
        assert_eq!(comms.size(), 2);
        assert_eq!(comms.devices(), &[Device::CUDA(0), Device::CUDA(1)]);
        // Drop cleans up
    }

    #[test]
    #[ignore = "NCCL needs 2+ GPUs; run with: make cuda-test-nccl"]
    fn test_nccl_broadcast() {
        if !require_multi_gpu() { return; }
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // Set values on device 0, zeros on device 1
        let t0 = Tensor::full(&[64], 42.0, opts0).unwrap();
        let t1 = Tensor::zeros(&[64], opts1).unwrap();

        // Broadcast from device 0
        comms.broadcast(&[&t0, &t1], 0).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "device 0 should still have 42.0");
        assert!(vals1.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "device 1 should have 42.0 after broadcast");
    }

    #[test]
    #[ignore = "NCCL needs 2+ GPUs; run with: make cuda-test-nccl"]
    fn test_nccl_all_reduce_sum() {
        if !require_multi_gpu() { return; }
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // 1.0 on device 0, 2.0 on device 1
        let t0 = Tensor::full(&[128], 1.0, opts0).unwrap();
        let t1 = Tensor::full(&[128], 2.0, opts1).unwrap();

        comms.all_reduce(&[&t0, &t1], ReduceOp::Sum).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);

        // Sum: 1.0 + 2.0 = 3.0 on both devices
        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 3.0).abs() < 1e-5),
            "device 0 should have 3.0 after AllReduce Sum");
        assert!(vals1.iter().all(|&v| (v - 3.0).abs() < 1e-5),
            "device 1 should have 3.0 after AllReduce Sum");
    }

    #[test]
    #[ignore = "NCCL needs 2+ GPUs; run with: make cuda-test-nccl"]
    fn test_nccl_all_reduce_avg() {
        if !require_multi_gpu() { return; }
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // 10.0 on device 0, 20.0 on device 1
        let t0 = Tensor::full(&[64], 10.0, opts0).unwrap();
        let t1 = Tensor::full(&[64], 20.0, opts1).unwrap();

        comms.all_reduce(&[&t0, &t1], ReduceOp::Avg).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);

        // Avg: (10.0 + 20.0) / 2 = 15.0
        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 15.0).abs() < 1e-5),
            "device 0 should have 15.0 after AllReduce Avg");
        assert!(vals1.iter().all(|&v| (v - 15.0).abs() < 1e-5),
            "device 1 should have 15.0 after AllReduce Avg");
    }

    #[test]
    #[ignore = "NCCL needs 2+ GPUs; run with: make cuda-test-nccl"]
    fn test_nccl_all_reduce_on_streams() {
        if !require_multi_gpu() { return; }
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        let stream0 = CudaStream::new(Device::CUDA(0), false).unwrap();
        let stream1 = CudaStream::new(Device::CUDA(1), false).unwrap();

        let t0 = Tensor::full(&[32], 5.0, opts0).unwrap();
        let t1 = Tensor::full(&[32], 7.0, opts1).unwrap();

        comms.all_reduce_on_streams(
            &[&t0, &t1], ReduceOp::Sum, &[&stream0, &stream1],
        ).unwrap();

        stream0.synchronize().unwrap();
        stream1.synchronize().unwrap();

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 12.0).abs() < 1e-5),
            "device 0 should have 12.0 after AllReduce Sum on streams");
        assert!(vals1.iter().all(|&v| (v - 12.0).abs() < 1e-5),
            "device 1 should have 12.0 after AllReduce Sum on streams");
    }
}
