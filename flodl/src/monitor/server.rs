//! Embedded HTTP server for the live training dashboard.
//!
//! Serves a self-contained HTML page at `/` and pushes epoch updates
//! via Server-Sent Events at `/events`. No external dependencies.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Sender, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Dashboard HTML — embedded at compile time.
const DASHBOARD_HTML: &str = include_str!("dashboard.html");

/// Messages from the Monitor to the server.
pub(crate) enum ServerMsg {
    /// New epoch data as JSON string.
    Epoch(String),
    /// Updated SVG graph.
    SetSvg(String),
    /// Clean shutdown.
    Shutdown,
}

/// A background HTTP server for the live training dashboard.
pub(crate) struct DashboardServer {
    tx: Sender<ServerMsg>,
    _accept_handle: JoinHandle<()>,
    msg_handle: Option<JoinHandle<()>>,
}

/// Shared state between handler threads.
struct SharedState {
    /// All epoch events seen so far (for catch-up on new SSE connections).
    epochs: Mutex<Vec<String>>,
    /// Current SVG graph.
    svg: Mutex<Option<String>>,
    /// SSE client senders — each connected SSE client has a channel.
    sse_senders: Mutex<Vec<Sender<String>>>,
}

impl DashboardServer {
    /// Start the dashboard server on the given port.
    pub fn start(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        let (tx, rx) = mpsc::channel::<ServerMsg>();

        let state = Arc::new(SharedState {
            epochs: Mutex::new(Vec::new()),
            svg: Mutex::new(None),
            sse_senders: Mutex::new(Vec::new()),
        });

        // Message handler thread: receives from Monitor, broadcasts to SSE clients
        let state2 = state.clone();
        let msg_handle = thread::spawn(move || {
            handle_messages(rx, state2);
        });

        // Acceptor thread: accepts TCP connections, spawns handler per connection
        let state3 = state.clone();
        let accept_handle = thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let state = state3.clone();
                thread::spawn(move || {
                    handle_connection(stream, &state);
                });
            }
        });

        Ok(Self {
            tx,
            _accept_handle: accept_handle,
            msg_handle: Some(msg_handle),
        })
    }

    /// Push an epoch update to all connected dashboard clients.
    pub fn push_epoch(&self, json: String) {
        let _ = self.tx.send(ServerMsg::Epoch(json));
    }

    /// Update the graph SVG.
    pub fn set_svg(&self, svg: String) {
        let _ = self.tx.send(ServerMsg::SetSvg(svg));
    }

    /// Signal shutdown and wait for the message handler to finish.
    pub fn shutdown(&mut self) {
        let _ = self.tx.send(ServerMsg::Shutdown);
        if let Some(h) = self.msg_handle.take() {
            let _ = h.join();
        }
    }
}

/// Process incoming messages from the Monitor.
fn handle_messages(rx: Receiver<ServerMsg>, state: Arc<SharedState>) {
    for msg in rx {
        match msg {
            ServerMsg::Epoch(json) => {
                let event = format!("event: epoch\ndata: {}\n\n", json);
                state.epochs.lock().unwrap().push(json);
                let mut senders = state.sse_senders.lock().unwrap();
                senders.retain(|tx| tx.send(event.clone()).is_ok());
            }
            ServerMsg::SetSvg(svg) => {
                *state.svg.lock().unwrap() = Some(svg);
            }
            ServerMsg::Shutdown => {
                let event = "event: complete\ndata: {}\n\n".to_string();
                let senders = state.sse_senders.lock().unwrap();
                for tx in senders.iter() {
                    let _ = tx.send(event.clone());
                }
                break;
            }
        }
    }
}

/// Handle a single HTTP connection.
fn handle_connection(mut stream: TcpStream, state: &SharedState) {
    let mut buf = [0u8; 2048];
    let n = stream.read(&mut buf).unwrap_or(0);
    if n == 0 {
        return;
    }

    let request = String::from_utf8_lossy(&buf[..n]);
    let path = parse_path(&request);

    match path {
        "/" => serve_html(&mut stream),
        "/events" => serve_sse(stream, state),
        "/graph.svg" => serve_svg(&mut stream, state),
        "/api/history" => serve_history(&mut stream, state),
        _ => {
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        }
    }
}

/// Extract the request path from the first line.
fn parse_path(request: &str) -> &str {
    request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
}

/// Serve the dashboard HTML.
fn serve_html(stream: &mut TcpStream) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
        DASHBOARD_HTML.len(),
        DASHBOARD_HTML,
    );
    let _ = stream.write_all(response.as_bytes());
}

/// Hold the connection open as an SSE stream.
fn serve_sse(mut stream: TcpStream, state: &SharedState) {
    let headers = "HTTP/1.1 200 OK\r\n\
                   Content-Type: text/event-stream\r\n\
                   Cache-Control: no-cache\r\n\
                   Connection: keep-alive\r\n\
                   Access-Control-Allow-Origin: *\r\n\r\n";
    if stream.write_all(headers.as_bytes()).is_err() {
        return;
    }

    // Send existing epochs as catch-up
    {
        let epochs = state.epochs.lock().unwrap();
        for json in epochs.iter() {
            let event = format!("event: epoch\ndata: {}\n\n", json);
            if stream.write_all(event.as_bytes()).is_err() {
                return;
            }
        }
        let _ = stream.flush();
    }

    // Register for future events
    let (tx, rx) = mpsc::channel::<String>();
    state.sse_senders.lock().unwrap().push(tx);

    // Block on receiving events until the client disconnects
    for event in rx {
        if stream.write_all(event.as_bytes()).is_err() {
            break;
        }
        let _ = stream.flush();
    }
}

/// Serve the current SVG graph.
fn serve_svg(stream: &mut TcpStream, state: &SharedState) {
    let svg = state.svg.lock().unwrap();
    if let Some(ref s) = *svg {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/svg+xml\r\nContent-Length: {}\r\n\r\n{}",
            s.len(),
            s,
        );
        let _ = stream.write_all(response.as_bytes());
    } else {
        let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
    }
}

/// Serve all epoch history as JSON (for late-connecting dashboards).
fn serve_history(stream: &mut TcpStream, state: &SharedState) {
    let epochs = state.epochs.lock().unwrap();
    let body = format!("[{}]", epochs.join(","));
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body,
    );
    let _ = stream.write_all(response.as_bytes());
}
