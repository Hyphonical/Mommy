# mommy

> [!NOTE]
> Current status: Alpha. Model has not been finetuned.

A tiny, pure-Rust embedded language model for shell prompt encouragement and compliments.

## Why & how it works

Sometimes `cargo build` hurts your feelings, and canned responses just don't cut it. Instead of picking from a fixed list of strings, `mommy` wraps your command, sees how it actually went (exit status, duration, and the last 32 KiB each of `stdout` and `stderr`) and writes you a fresh, context-aware little note of praise or comfort.

Everything runs locally inside a pure-Rust single binary on Windows, Linux, and macOS. There are no API keys, no background servers, no Python environments, and no C/C++ runtime dependencies.

## Installation

```sh
cargo install mommy
```

On the very first run, `mommy` downloads its ~128 MB model file once and caches it in your user data directory.

## Quick start

Put `mommy` in front of whatever you're running (use `--` if you need to separate flags):

```sh
mommy cargo test
mommy -- git push origin main
```

Illustrative terminal session:

```text
$ mommy cargo test
   Compiling demo v0.1.0
    Finished `test` profile [unoptimized + debuginfo] target(s) in 1.4s
     Running unittests src/lib.rs
test result: ok. 12 passed; 0 failed; 0 ignored

all twelve tests green, mommy knew you had it in you~

$ mommy cargo check
error[E0382]: borrow of moved value: `buf`
...
the borrow checker can be picky, take a breath and try again for mommy~
```

`mommy` always exits with the wrapped command's exact exit code. When `stderr` is not a TTY (such as in CI or shell pipelines), `mommy` stays silent and passes the command straight through; set `MOMMY_FORCE=1` if you want encouragement anyway.

## Shell integration

Wrapping a command directly (`mommy <command>`) gives the model full context, including output tails and timing. If you want `mommy` to chime in automatically after every prompt, these illustrative hook snippets forward the previous command's exit status (without its output or duration):

**Bash** (`~/.bashrc`):
```bash
PROMPT_COMMAND='mommy -- sh -c "exit $?"'
```

**Zsh** (`~/.zshrc`):
```zsh
precmd() { mommy -- sh -c "exit $?"; }
```

**Fish** (`~/.config/fish/config.fish`):
```fish
function __mommy --on-event fish_postexec; mommy -- sh -c "exit $status"; end
```

**Nushell** (`config.nu`):
```nu
$env.config.hooks.pre_prompt = [{ || mommy -- nu -c $"exit ($env.LAST_EXIT_CODE)" }]
```

**PowerShell** (`$PROFILE`):
```powershell
function prompt { $c = $LASTEXITCODE; mommy -- cmd /c "exit $c"; "PS $($executionContext.SessionState.Path.CurrentLocation)> " }
```

## Configuration

Config lives at `~/.mommy/config.toml`, or at the path in `$MOMMY_CONFIG` if set.

Persona names (`char`, `user`, `them`, `their`, `self`) can be a single name or a list. One name is picked at random each time, and that same pick is used in the whole message. An empty list uses the default.

```toml
[persona]
user = ["honey", "baby", "sweetie"]
# or a single name
# user = "sweetie"
```

Defaults are `mommy`, `good girl`, `her`, `her` and `herself`.

`style.color` takes a hex colour with a hash, like `"#ff99cc"`. It colours the output only if your terminal supports colour, and is turned off if `NO_COLOR` is set. A bad colour prints a warning and falls back to plain text.

```toml
[style]
color = "#ff99cc"
```

## Single-binary build (`embed` feature)

If you want a completely self-contained executable with no first-run download, place the quantized weights at `model/model.bpk` and enable the `embed` feature:

```sh
cargo build --release --features embed
```

This bakes `model/model.bpk` directly into the binary at compile time via `include_bytes!`. If you are working from raw weights locally, `mommy --convert` runs a one-shot weight conversion utility. (Note: `embed` is intentionally off by default and must not be enabled when publishing to crates.io.)

## Model & download notes

Under the hood is a fine-tuned 65.1M-parameter Qwen3 model (`SupraLabs/Supra2.5-Pro-Instruct` base) using ChatML formatting and a 16,384-token ByteLevel BPE vocabulary via Hugging Face `tokenizers`. Inference runs in pure Rust on `burn` (Flex SIMD backend + Rayon), sampling at temperature `0.8` and top-k `40`, stopping at `~`, a newline, or 64 tokens.

To keep the crates.io package small, weights are not bundled in the crate. Default builds download `model.bpk` (~128 MB) on first launch from https://huggingface.co/Hyphonical/Mommy into your OS user data directory and reuse that cached copy offline from then on.

## Limitations

- **First-run download:** Default installs need ~128 MB of disk space and an initial download before offline use.
- **CPU-only inference:** Everything runs on your CPU via SIMD and Rayon with no GPU backend, so there is a brief pause after each command while `mommy` thinks.
- **Tiny model quirks:** With 65.1M parameters, a small 16k BPE vocabulary, and a 512-token context window, verbose logs get truncated and phrasing can occasionally get a little weird.

## License & credits

Licensed under `AGPL-3.0-only` — see [LICENSE](LICENSE).

Inspired by `fwdekker/mommy`, and built by fine-tuning `SupraLabs/Supra2.5-Pro-Instruct`.