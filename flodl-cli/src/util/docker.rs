//! Docker and Docker Compose interaction.

use std::process::{Command, ExitStatus, Output, Stdio};

/// Check whether Docker is available and the daemon is running.
pub fn has_docker() -> bool {
    Command::new("docker")
        .arg("info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Check whether a Docker image exists locally.
#[allow(dead_code)]
pub fn image_exists(name: &str) -> bool {
    Command::new("docker")
        .args(["image", "inspect", name])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Run `docker compose <args>` with stdout/stderr inherited (streaming).
///
/// Returns the exit status. The `compose_dir` is set as the working directory.
#[allow(dead_code)]
pub fn compose_run(compose_dir: &str, args: &[&str]) -> Result<ExitStatus, String> {
    Command::new("docker")
        .arg("compose")
        .args(args)
        .current_dir(compose_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("failed to run docker compose: {}", e))
}

/// Run `docker compose <args>` and capture output.
#[allow(dead_code)]
pub fn compose_output(compose_dir: &str, args: &[&str]) -> Result<Output, String> {
    Command::new("docker")
        .arg("compose")
        .args(args)
        .current_dir(compose_dir)
        .output()
        .map_err(|e| format!("failed to run docker compose: {}", e))
}

/// Run `docker <args>` and capture output.
pub fn docker_output(args: &[&str]) -> Result<Output, String> {
    Command::new("docker")
        .args(args)
        .output()
        .map_err(|e| format!("failed to run docker: {}", e))
}

/// Run `docker <args>` with streaming output.
pub fn docker_run(args: &[&str]) -> Result<ExitStatus, String> {
    Command::new("docker")
        .args(args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("failed to run docker: {}", e))
}
