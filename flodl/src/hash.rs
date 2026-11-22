//! SHA-256 and HMAC-SHA256, the one place the crate names its hashing
//! backend.
//!
//! The backend picks the CPU's SHA extensions at run time when it has
//! them. That matters for the CPU-averaging reduce, which authenticates a
//! model-sized frame at every hop: on a CPU with the extensions a pass
//! over a frame is several times faster than a portable implementation.

use hmac::{Hmac, KeyInit, Mac};
use sha2::Digest;

/// Incremental SHA-256.
pub(crate) struct Sha256(sha2::Sha256);

impl Sha256 {
    pub(crate) fn new() -> Self {
        Sha256(sha2::Sha256::new())
    }

    pub(crate) fn update(&mut self, bytes: impl AsRef<[u8]>) {
        self.0.update(bytes.as_ref());
    }

    pub(crate) fn finalize(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

/// Incremental HMAC-SHA256 (RFC 2104). Feeding a message in several
/// updates gives the same tag as feeding their concatenation, so a caller
/// with a header and a payload never has to join them first.
pub(crate) struct HmacSha256(Hmac<sha2::Sha256>);

impl HmacSha256 {
    pub(crate) fn new(key: &[u8]) -> Self {
        // HMAC takes a key of any length; only a fixed-size MAC can
        // reject one.
        HmacSha256(Hmac::new_from_slice(key).expect("HMAC accepts a key of any length"))
    }

    pub(crate) fn update(&mut self, bytes: impl AsRef<[u8]>) {
        self.0.update(bytes.as_ref());
    }

    pub(crate) fn finalize(self) -> [u8; 32] {
        self.0.finalize().into_bytes().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha256_matches_the_standard_vector() {
        // FIPS 180-2, "abc".
        let mut h = Sha256::new();
        h.update(b"abc");
        assert_eq!(
            hex(&h.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hmac_matches_rfc_4231() {
        // Test case 2: a key shorter than the block.
        let mut m = HmacSha256::new(b"Jefe");
        m.update(b"what do ya want for nothing?");
        assert_eq!(
            hex(&m.finalize()),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // Test case 6: a key longer than the block, which HMAC hashes first.
        let mut m = HmacSha256::new(&[0xaa; 131]);
        m.update(b"Test Using Larger Than Block-Size Key - Hash Key First");
        assert_eq!(
            hex(&m.finalize()),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn split_updates_equal_one_update() {
        let mut whole = HmacSha256::new(b"key");
        whole.update(b"header-payload");
        let mut parts = HmacSha256::new(b"key");
        parts.update(b"header-");
        parts.update(b"payload");
        assert_eq!(whole.finalize(), parts.finalize());
    }
}
