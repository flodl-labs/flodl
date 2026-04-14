//! Standard dataset parsers (MNIST, CIFAR-10, Shakespeare).
//!
//! Pure parsers: accept raw bytes, return tensors. No I/O or download logic.
//! Implement [`super::BatchDataSet`] for direct use with [`super::DataLoader`].

pub mod mnist;
pub mod cifar10;
pub mod shakespeare;

pub use mnist::Mnist;
pub use cifar10::Cifar10;
pub use shakespeare::Shakespeare;
