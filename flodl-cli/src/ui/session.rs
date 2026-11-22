//! The session: who may talk to this server, and how a browser that lost
//! its way gets back in.
//!
//! The token is minted at startup and travels in the URL `fdl ui` prints.
//! Opening that URL trades it for a cookie, and the cookie is what every
//! route checks. The URL stays valid for the server's lifetime, so a
//! second browser, or one whose cookies were cleared, opens it again.
//!
//! A long run outlives a terminal's scrollback, so the URL is also kept
//! in a file only its owner can read, which `fdl ui --url` prints. Owner
//! only is the same boundary the token draws: another local user can
//! reach the loopback port, and cannot read this file.

use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Cookie name, per port: cookies are scoped by host and not by port, so
/// two servers on one box would otherwise overwrite each other's session.
pub(super) fn cookie_name(port: u16) -> String {
    format!("fdl_ui_{port}")
}

/// The URL that opens a session.
pub(super) fn login_url(port: u16, token: &str) -> String {
    format!("http://127.0.0.1:{port}/?token={token}")
}

/// One named value out of a `Cookie:` header.
pub(super) fn cookie_value<'a>(header: &'a str, name: &str) -> Option<&'a str> {
    header.split(';').find_map(|pair| {
        let (k, v) = pair.trim().split_once('=')?;
        (k == name).then_some(v)
    })
}

/// Compare a presented token with the session's without stopping at the
/// first differing byte.
pub(super) fn token_matches(presented: &str, token: &str) -> bool {
    let (a, b) = (presented.as_bytes(), token.as_bytes());
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn session_dir(root: &Path) -> PathBuf {
    root.join(".fdl").join("ui")
}

fn session_path(root: &Path, port: u16) -> PathBuf {
    session_dir(root).join(format!("session-{port}.json"))
}

/// Record this server's URL for `fdl ui --url`.
///
/// On Unix the file is created owner-only, before anything is written to
/// it. Windows has no mode bits: there the file is as private as the
/// project directory's own access list makes it.
pub(super) fn write(root: &Path, port: u16, token: &str) -> std::io::Result<PathBuf> {
    use std::io::Write as _;
    let dir = session_dir(root);
    std::fs::create_dir_all(&dir)?;
    // `.fdl/` holds credentials and ignores itself; make sure of it here,
    // since this may be the first thing a project writes there.
    let ignore = root.join(".fdl").join(".gitignore");
    if !ignore.exists() {
        std::fs::write(&ignore, "*\n")?;
    }
    let path = session_path(root, port);
    // A leftover from an earlier server on this port may carry other
    // permissions; start from nothing.
    let _ = std::fs::remove_file(&path);
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        opts.mode(0o600);
    }
    let mut f = opts.open(&path)?;
    let body = serde_json::json!({
        "port": port,
        "pid": std::process::id(),
        "url": login_url(port, token),
    });
    writeln!(f, "{body}")?;
    Ok(path)
}

/// URLs of the servers still answering for this project, by port. A
/// record whose port no longer accepts a connection is a server that was
/// killed, and is removed.
pub(super) fn live_urls(root: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(session_dir(root)) else {
        return Vec::new();
    };
    let mut found: Vec<(u16, String)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.starts_with("session-") && name.ends_with(".json")) {
            continue;
        }
        let record = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let parsed = record.as_ref().and_then(|v| {
            let port = u16::try_from(v["port"].as_u64()?).ok()?;
            Some((port, v["url"].as_str()?.to_string()))
        });
        match parsed {
            Some((port, url)) if answers(port) => found.push((port, url)),
            _ => {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    found.sort();
    found.into_iter().map(|(_, url)| url).collect()
}

fn answers(port: u16) -> bool {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_ok()
}

/// What a browser without a session gets at `/`: how to get one, and a
/// field for the token itself.
pub(super) fn login_page(note: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
<title>fdl ui</title><style>\
body{{font:16px/1.5 system-ui,sans-serif;max-width:34rem;margin:15vh auto;padding:0 1rem}}\
code{{background:#8882;padding:.1em .3em;border-radius:3px}}\
input{{font:inherit;padding:.4em;width:22rem;max-width:100%}}\
button{{font:inherit;padding:.4em .9em}}.note{{color:#b45309}}\
</style></head><body><h1>fdl ui</h1>\
<p class=\"note\">{note}</p>\
<p>This page opens with the URL <code>fdl ui</code> printed when it started. \
If that is out of reach, run <code>fdl ui --url</code> in the project on the box \
that serves it, and open what it prints.</p>\
<form method=\"get\" action=\"/\"><p><label>Or paste the token (the part after \
<code>token=</code>):<br><input name=\"token\" autocomplete=\"off\" autofocus></label> \
<button>Open</button></p></form></body></html>"
    )
}
