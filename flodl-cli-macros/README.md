# flodl-cli-macros

Proc-macro derive crate for [`flodl-cli`](https://crates.io/crates/flodl-cli),
the `fdl` command-line tool for the [floDl](https://flodl.dev) Rust deep
learning framework.

This crate exposes a single derive, `#[derive(FdlArgs)]`, which turns a
plain struct into an argv parser that plugs into `flodl-cli`'s runtime.
It is re-exported by `flodl-cli` as `flodl_cli::FdlArgs`, so depending on
`flodl-cli` is enough. You do not normally add this crate directly.

## What the derive produces

For a struct annotated with `#[derive(FdlArgs)]`, the macro implements
`flodl_cli::FdlArgsTrait`, which provides:

- `try_parse_from(&[String]) -> Result<Self, String>` — parses argv, with
  optional `env` fallback per field and typed `FromStr` conversion.
- `schema() -> flodl_cli::Schema` — JSON-serialisable description of
  every flag and positional, used for `--fdl-schema` and shell
  completions.
- `render_help() -> String` — ANSI-coloured `--help` text assembled from
  doc-comments and attribute metadata.

## Attributes

Each field must carry exactly one of `#[option(...)]` or `#[arg(...)]`.

### `#[option]` (named flag, kebab-cased from the field ident)

| Key         | Value            | Notes                                                     |
|-------------|------------------|-----------------------------------------------------------|
| `short`     | `'c'`            | Single-char short flag.                                   |
| `default`   | `"string"`       | Parsed via `FromStr` at run time; required on bare `T`.   |
| `choices`   | `&["a", "b"]`    | Accepted values, enforced by the parser.                  |
| `env`       | `"VAR_NAME"`     | Env fallback when the flag is absent; skipped on `bool`.  |
| `completer` | `"name"`         | Named completer for shell completions.                    |

Supported field shapes:

- `bool` — absent = `false`, present = `true`.
- `T` — scalar, requires `default = "..."`.
- `Option<T>` — absent = `None`.
- `Vec<T>` — repeatable flag, collects all values.

### `#[arg]` (positional)

| Key         | Value            | Notes                                                     |
|-------------|------------------|-----------------------------------------------------------|
| `default`   | `"string"`       | Makes the arg optional.                                   |
| `choices`   | `&["a", "b"]`    | Accepted values.                                          |
| `variadic`  | bare or `= true` | Requires `Vec<T>`; must be the last positional.           |
| `completer` | `"name"`         | Named completer for shell completions.                    |

Validation is enforced at derive time: required positionals cannot
follow optional ones, variadic must be last, reserved flags
(`--help`, `--version`, `--quiet`, `--env`, `-h`, `-V`, `-q`, `-v`,
`-e`) cannot be shadowed, and duplicate long/short flags error out.

## Example

```rust
use flodl_cli::FdlArgs;
use std::path::PathBuf;

/// Run a training job.
#[derive(FdlArgs, Debug)]
pub struct TrainArgs {
    /// Path to the config file.
    #[arg]
    pub config: PathBuf,

    /// Number of epochs.
    #[option(short = 'e', default = "10")]
    pub epochs: u32,

    /// Device to train on.
    #[option(choices = &["cpu", "cuda"], default = "cuda")]
    pub device: String,

    /// API key (falls back to env).
    #[option(env = "FLODL_API_KEY")]
    pub api_key: Option<String>,

    /// Extra dataset paths.
    #[arg(variadic)]
    pub extra: Vec<PathBuf>,
}
```

With this in place, `TrainArgs::try_parse_from(&argv)` yields a typed
struct, `TrainArgs::render_help()` produces the `--help` banner, and
`TrainArgs::schema()` feeds `fdl`'s completion and introspection
pipeline. See the [flodl-cli README](https://crates.io/crates/flodl-cli)
for the wider CLI surface this plugs into.

## License

floDl is open-sourced software licensed under the [MIT license](https://github.com/fab2s/floDl/blob/main/LICENSE).
