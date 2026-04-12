//! CIFAR-10 dataset parser.
//!
//! Parses the binary batch format into tensors.
//! Images are normalized to [0, 1] as Float32, labels are Int64.
//!
//! Each batch file contains 10,000 images in the format:
//! `[1 byte label][1024 R pixels][1024 G pixels][1024 B pixels]` per image.
//!
//! # Example
//!
//! ```ignore
//! let batch1 = std::fs::read("data_batch_1.bin")?;
//! let batch2 = std::fs::read("data_batch_2.bin")?;
//! // ... load all 5 training batches
//! let cifar = Cifar10::parse(&[&batch1, &batch2, ...])?;
//! // cifar.images: [50000, 3, 32, 32] Float32
//! // cifar.labels: [50000] Int64
//! ```

use crate::data::BatchDataSet;
use crate::tensor::{Device, Result, Tensor, TensorError};

/// CIFAR-10 class names in label order.
pub const CLASS_NAMES: [&str; 10] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
];

const PIXELS_PER_IMAGE: usize = 3 * 32 * 32; // 3072
const BYTES_PER_RECORD: usize = 1 + PIXELS_PER_IMAGE; // 3073
const IMAGES_PER_BATCH: usize = 10_000;

/// Parsed CIFAR-10 dataset.
pub struct Cifar10 {
    /// Images as `[N, 3, 32, 32]` Float32, normalized to [0, 1].
    pub images: Tensor,
    /// Labels as `[N]` Int64 (0-9).
    pub labels: Tensor,
}

impl Cifar10 {
    /// Parse one or more raw CIFAR-10 binary batch files.
    ///
    /// Each slice should be the raw (uncompressed) contents of a batch file
    /// (e.g. `data_batch_1.bin`). Pass all 5 training batches for the full
    /// 50,000-image training set, or the single test batch for 10,000 test images.
    pub fn parse(batches: &[&[u8]]) -> Result<Self> {
        if batches.is_empty() {
            return Err(TensorError::new("CIFAR-10: no batch data provided"));
        }

        let mut all_pixels: Vec<f32> = Vec::new();
        let mut all_labels: Vec<i64> = Vec::new();

        for (batch_idx, &batch) in batches.iter().enumerate() {
            let expected = IMAGES_PER_BATCH * BYTES_PER_RECORD;
            if batch.len() != expected {
                return Err(TensorError::new(&format!(
                    "CIFAR-10 batch {}: expected {} bytes, got {}",
                    batch_idx, expected, batch.len()
                )));
            }

            for img_idx in 0..IMAGES_PER_BATCH {
                let offset = img_idx * BYTES_PER_RECORD;
                let label = batch[offset] as i64;
                if label > 9 {
                    return Err(TensorError::new(&format!(
                        "CIFAR-10 batch {} image {}: invalid label {}",
                        batch_idx, img_idx, label
                    )));
                }
                all_labels.push(label);

                // Pixels are already in CHW order: [1024 R][1024 G][1024 B]
                let pixel_start = offset + 1;
                let pixel_end = pixel_start + PIXELS_PER_IMAGE;
                for &b in &batch[pixel_start..pixel_end] {
                    all_pixels.push(b as f32 / 255.0);
                }
            }
        }

        let n = all_labels.len() as i64;
        let images = Tensor::from_f32(&all_pixels, &[n, 3, 32, 32], Device::CPU)?;
        let labels = Tensor::from_i64(&all_labels, &[n], Device::CPU)?;

        Ok(Cifar10 { images, labels })
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.images.shape()[0] as usize
    }

    /// True if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl BatchDataSet for Cifar10 {
    fn len(&self) -> usize {
        self.images.shape()[0] as usize
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        let idx: Vec<i64> = indices.iter().map(|&i| (i % self.len()) as i64).collect();
        let idx_tensor = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;
        let images = self.images.index_select(0, &idx_tensor)?;
        let labels = self.labels.index_select(0, &idx_tensor)?;
        Ok(vec![images, labels])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal CIFAR-10 batch with `n` images.
    fn make_batch(n: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(n * BYTES_PER_RECORD);
        for i in 0..n {
            buf.push((i % 10) as u8); // label
            // R channel: all (i % 256)
            for _ in 0..1024 {
                buf.push((i % 256) as u8);
            }
            // G channel: all 0
            buf.extend_from_slice(&[0u8; 1024]);
            // B channel: all 255
            buf.extend_from_slice(&[255u8; 1024]);
        }
        buf
    }

    #[test]
    fn parse_single_batch() {
        let batch = make_batch(IMAGES_PER_BATCH);
        let cifar = Cifar10::parse(&[&batch]).unwrap();

        assert_eq!(cifar.images.shape(), &[10000, 3, 32, 32]);
        assert_eq!(cifar.labels.shape(), &[10000]);

        // First image label = 0
        let l = cifar.labels.select(0, 0).unwrap().to_i64_vec().unwrap()[0];
        assert_eq!(l, 0);

        // Second image label = 1
        let l = cifar.labels.select(0, 1).unwrap().to_i64_vec().unwrap()[0];
        assert_eq!(l, 1);
    }

    #[test]
    fn parse_multiple_batches() {
        let b1 = make_batch(IMAGES_PER_BATCH);
        let b2 = make_batch(IMAGES_PER_BATCH);
        let cifar = Cifar10::parse(&[&b1, &b2]).unwrap();
        assert_eq!(cifar.images.shape(), &[20000, 3, 32, 32]);
    }

    #[test]
    fn wrong_size_rejected() {
        let batch = [0u8; 100]; // way too short
        assert!(Cifar10::parse(&[&batch[..]]).is_err());
    }

    #[test]
    fn pixel_normalization() {
        let batch = make_batch(IMAGES_PER_BATCH);
        let cifar = Cifar10::parse(&[&batch]).unwrap();

        // Image 0: R channel all 0 -> 0.0
        let img0 = cifar.images.select(0, 0).unwrap();
        let r_pixel: f64 = img0.select(0, 0).unwrap() // R channel
            .select(0, 0).unwrap() // row 0
            .select(0, 0).unwrap() // col 0
            .item().unwrap();
        assert!((r_pixel - 0.0).abs() < 1e-6);

        // B channel all 255 -> 1.0
        let b_pixel: f64 = img0.select(0, 2).unwrap() // B channel
            .select(0, 0).unwrap()
            .select(0, 0).unwrap()
            .item().unwrap();
        assert!((b_pixel - 1.0).abs() < 1e-6);
    }
}
