//! The cloud-init user-data variant: an instance that boots straight
//! into `fdl join`.
//!
//! A SECRET artifact from the moment it exists (private key + admission
//! token inside), which is why it lands 0600 in the farm dir and never
//! on stdout. The systemd unit encodes the failure taxonomy end to end:
//! transient exits re-dial, permanent ones stop the loop and halt.

use std::path::Path;

use crate::build_info::BuildInfo;

use super::Door;

/// How an instance gets its `fdl`.
///
/// It has to be the fdl the farm was configured with: a walk-in's fdl
/// dials, pulls and reads the run manifest the controller's side
/// produces, so the published release serves only a release build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FdlInstall {
    /// The published release, through the `flodl.dev/fdl` bootstrap.
    Release,
    /// `cargo install` of the configuring fdl's own commit.
    Commit { repo: String, commit: String },
}

/// The install an instance gets, from the configuring fdl's identity.
///
/// A release build hands out the release. Any other build with a
/// commit hands out that commit, which must be pushed to `repo` before
/// an instance boots. A dirty build is refused: no remote can install
/// uncommitted sources, so a pin would install a different fdl than the
/// one that configured the farm, which is the mismatch this exists to
/// prevent.
pub(super) fn fdl_install(build: &BuildInfo, repo: &str) -> Result<FdlInstall, String> {
    if build.is_release() {
        return Ok(FdlInstall::Release);
    }
    let commit = build.commit.unwrap_or_default();
    if build.dirty {
        return Err(format!(
            "this fdl was built from uncommitted sources ({}): an instance \
             cannot install them, so cloud-init would boot a different fdl \
             than the one configuring this farm. Commit and push, rebuild \
             fdl, then re-run with --cloud-init.",
            build.version_line()
        ));
    }
    Ok(FdlInstall::Commit {
        repo: repo.to_string(),
        commit: commit.to_string(),
    })
}

pub(super) fn home_of(user: &str) -> String {
    if user == "root" {
        "/root".to_string()
    } else {
        format!("/home/{user}")
    }
}

/// The cloud-init user-data: worker yml, private key, the tools the
/// declared door needs and the systemd recipe, so an instance boots
/// straight into a persistent `fdl join`. A SECRET artifact from the
/// moment it exists — it carries the key and the token — which is why
/// it lands 0600 in the farm dir and never on stdout.
///
/// The unit encodes the failure taxonomy end to end: `Restart=always`
/// re-dials transient exits, `RestartPreventExitStatus=2` stops the hot
/// loop on a permanent one, and `FailureAction=poweroff` (a `[Unit]`
/// option) then halts the instance.
///
/// **Halting is not always deprovisioning.** On providers that keep
/// billing a powered-off instance (DigitalOcean, and the AMD Developer
/// Cloud that runs on it, reserve disk/CPU/RAM/IP until the instance is
/// destroyed), the unit stops the work but not the meter. There, pair
/// it with a provider-side destroy.
///
/// What the instance is assumed to have is only a shell, systemd and
/// network: `fdl` is fetched here, and the door's own tooling with it.
/// Tooling already baked into the image wins, since those steps are
/// guarded by a `command -v`. A commit-pinned fdl is not: an image's own
/// fdl is exactly the one that would not match.
pub(super) fn render_cloud_init(
    label: &str,
    user: &str,
    door: Door,
    worker_yml: &str,
    private_key: &str,
    fdl: &FdlInstall,
) -> String {
    let indent = |s: &str| -> String {
        s.lines()
            .map(|l| {
                if l.is_empty() {
                    String::new()
                } else {
                    format!("      {l}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let home = home_of(user);

    // Packages the DECLARED door will actually reach for. Door B pulls a
    // source tree and builds it here, so it needs a compiler and the
    // transports; door A mounts the data root over sshfs, and prepare
    // classes a missing sshfs as permanent — which under this very unit
    // means exit 2 and a halt, on a box that only lacked a package.
    let pinned = matches!(fdl, FdlInstall::Commit { .. });
    let mut packages: Vec<&str> = vec!["curl"];
    match door {
        Door::B => packages.extend(["build-essential", "pkg-config", "unzip", "rsync", "git"]),
        Door::A => packages.push("sshfs"),
        Door::Nologin => {}
    }
    // A pinned fdl is compiled here, whatever the door: it needs a linker.
    if pinned && !packages.contains(&"build-essential") {
        packages.push("build-essential");
    }
    let packages = packages
        .iter()
        .map(|p| format!("\x20 - {p}\n"))
        .collect::<String>();

    // Door B builds, and so does a pinned fdl on any door, so both need a
    // toolchain. Installed AS THE SERVICE USER: cargo writes its registry
    // cache into CARGO_HOME, so a system-wide install root-owned and
    // world-readable is a build that fails on its first fetch. The unit
    // then carries the matching PATH rather than relying on a login shell
    // it never gets, and a pinned fdl lands in that same bin, first on it.
    let (rust_step, rust_path) = if door == Door::B || pinned {
        (
            format!(
                "\x20 - [ sh, -c, \"command -v cargo >/dev/null || \
                 su -l {user} -c 'curl -fsSL https://sh.rustup.rs | \
                 sh -s -- -y --profile minimal --no-modify-path'\" ]\n"
            ),
            format!("{home}/.cargo/bin:"),
        )
    } else {
        (String::new(), String::new())
    };

    // The release is fetched before the toolchain (it needs none); a
    // commit is compiled after it.
    let (fdl_comment, fdl_step, setup_steps) = match fdl {
        FdlInstall::Release => (
            "# fdl: the published release (flodl.dev/fdl).\n".to_string(),
            String::new(),
            format!(
                "\x20 - [ sh, -c, \"command -v fdl >/dev/null || \
                 (curl -fsSL https://flodl.dev/fdl -o /usr/local/bin/fdl && \
                 chmod 0755 /usr/local/bin/fdl)\" ]\n\
                 {rust_step}"
            ),
        ),
        FdlInstall::Commit { repo, commit } => (
            format!("# fdl: commit {commit} of {repo}, the configuring fdl's own.\n"),
            format!(
                "\x20 - [ sh, -c, \"su -l {user} -c '{home}/.cargo/bin/cargo install \
                 --locked --git {repo} --rev {commit} flodl-cli'\" ]\n"
            ),
            rust_step,
        ),
    };

    format!(
        "#cloud-config\n\
         # Farm `{label}` worker user-data — generated by `fdl join-config`.\n\
         # SECRET ARTIFACT: carries the join key and the admission token.\n\
         # On a provider that bills powered-off instances (DigitalOcean and\n\
         # the AMD Developer Cloud on top of it), the unit's poweroff stops\n\
         # the work but NOT the meter: destroy the instance to stop billing.\n\
         {fdl_comment}\
         packages:\n{packages}\
         write_files:\n\
         \x20 - path: {home}/.ssh/flodl-join\n\
         \x20   owner: {user}:{user}\n\
         \x20   permissions: \"0600\"\n\
         \x20   defer: true\n\
         \x20   content: |\n{key}\n\
         \x20 - path: {home}/training/fdl.yml\n\
         \x20   owner: {user}:{user}\n\
         \x20   permissions: \"0644\"\n\
         \x20   defer: true\n\
         \x20   content: |\n{yml}\n\
         \x20 - path: /etc/systemd/system/flodl-join.service\n\
         \x20   permissions: \"0644\"\n\
         \x20   content: |\n\
         \x20     [Unit]\n\
         \x20     Description=flodl walk-in worker (farm {label})\n\
         \x20     After=network-online.target\n\
         \x20     Wants=network-online.target\n\
         \x20     FailureAction=poweroff\n\
         \n\
         \x20     [Service]\n\
         \x20     Type=simple\n\
         \x20     User={user}\n\
         \x20     WorkingDirectory={home}/training\n\
         \x20     Environment=PATH={rust_path}/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n\
         \x20     ExecStart=/usr/bin/env fdl join\n\
         \x20     Restart=always\n\
         \x20     RestartSec=5\n\
         \x20     RestartPreventExitStatus=2\n\
         \n\
         \x20     [Install]\n\
         \x20     WantedBy=multi-user.target\n\
         runcmd:\n\
         {setup_steps}\
         {fdl_step}\
         \x20 - systemctl daemon-reload\n\
         \x20 - systemctl enable --now flodl-join.service\n",
        label = label,
        user = user,
        home = home,
        packages = packages,
        fdl_comment = fdl_comment,
        setup_steps = setup_steps,
        fdl_step = fdl_step,
        rust_path = rust_path,
        key = indent(private_key),
        yml = indent(worker_yml),
    )
}

/// The compose services this project dispatches commands through, if
/// any, from the merged config's `docker:` keys.
///
/// This is the disambiguator between the two dockerized shapes fdl runs
/// in. Inside a container, `/.dockerenv` answers. On the HOST, nothing
/// about the process says so — but a project whose commands carry
/// `docker: <service>` builds and runs THERE, so a tool this box lacks
/// may need installing in the image rather than here, and a fix aimed at
/// the host would be aimed at the wrong machine.
pub(super) fn docker_services(root: &Path, label: &str) -> Vec<String> {
    let Some(base) = crate::config::find_project_config(root) else {
        return Vec::new();
    };
    let Ok(project) = crate::config::load_project_with_env(&base, Some(label)) else {
        return Vec::new();
    };
    let mut seen: Vec<String> = Vec::new();
    for spec in project.commands.values() {
        if let Some(svc) = &spec.docker
            && !seen.contains(svc)
        {
            seen.push(svc.clone());
        }
    }
    seen
}
