#[cfg(not(feature = "embed"))]
mod fetch;
mod chat;
mod model {
    pub(crate) mod qwen3;
}
mod rng;
mod tokenizer;

use std::env;
use std::io::{self, IsTerminal, Read, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::TensorData;

use model::qwen3::{KvCache, Qwen3Config, Qwen3Model};
use tokenizer::Tokenizer;

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

const CONTEXT_TOKENS: usize = 128;
const STDOUT_TOKENS: usize = 32;
const GENERATION_CAP: usize = 64;
const MODEL_CONTEXT: usize = 512;
const TAIL_BYTES: usize = 32 * 1024;
const TEE_GRACE: Duration = Duration::from_millis(200);
const TEMPERATURE: f32 = 0.8;
const TOP_K: usize = 40;
const USAGE: &str = "usage: mommy [--] <command> [args...]";

fn main() {
    std::process::exit(real_main());
}

fn real_main() -> i32 {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--") {
        args.remove(0);
    }
    if args.first().map(String::as_str) == Some("--convert") {
        return run_convert();
    }
    let Some(command) = args.first().cloned() else {
        eprintln!("{USAGE}");
        return 2;
    };
    let cmd_args = args[1..].to_vec();

    let forced = env::var("MOMMY_FORCE").is_ok_and(|value| value == "1");
    if !io::stderr().is_terminal() && !forced {
        return run_inherited(&command, &cmd_args);
    }
    run_wrapped(&command, &cmd_args)
}

fn run_convert() -> i32 {
    let result: anyhow::Result<()> = (|| {
        let device = Device::default();
        let config = Qwen3Config::parse(include_str!("assets/config.json"))?;
        let model = Qwen3Model::load(config, model_path("model.safetensors"), &device)?;
        let output = model_path("model.bpk");
        model.save_bpk(&output)?;
        let bytes = std::fs::metadata(&output)?.len();
        eprintln!("wrote {} ({bytes} bytes)", output.display());
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(error) => {
            warn(&error);
            1
        }
    }
}

fn run_inherited(command: &str, args: &[String]) -> i32 {
    let mut child = match base_command(command, args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            eprintln!("mommy: failed to spawn {command}: {error}");
            return spawn_error_code(&error);
        }
    };
    match child.wait() {
        Ok(status) => status_exit_code(&status),
        Err(error) => {
            eprintln!("mommy: failed to wait for {command}: {error}");
            1
        }
    }
}

fn run_wrapped(command: &str, args: &[String]) -> i32 {
    let start = Instant::now();
    let device = Device::default();
    let full_command = join_command(command, args);

    let mut child = match base_command(command, args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            let message = spawn_failure_message(command, &error);
            eprintln!("{message}");
            let exit_code = spawn_error_code(&error);
            let load_result = load_model(&device);
            respond(
                load_result,
                &device,
                &GenerateInput {
                    status: exit_code,
                    command: &full_command,
                    elapsed: start.elapsed(),
                    stdout_tail: b"",
                    stderr_tail: message.as_bytes(),
                },
            );
            return exit_code;
        }
    };

    let stdout_handle = child
        .stdout
        .take()
        .map(|stream| thread::spawn(move || tee(stream, io::stdout())));
    let stderr_handle = child
        .stderr
        .take()
        .map(|stream| thread::spawn(move || tee(stream, io::stderr())));

    let load_result = load_model(&device);

    let status = match child.wait() {
        Ok(status) => status,
        Err(error) => {
            eprintln!("mommy: failed to wait for {command}: {error}");
            return 1;
        }
    };
    let elapsed = start.elapsed();
    let exit_code = status_exit_code(&status);

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let stdout_tail = stdout_handle
            .map(|handle| handle.join().unwrap_or_default())
            .unwrap_or_default();
        let stderr_tail = stderr_handle
            .map(|handle| handle.join().unwrap_or_default())
            .unwrap_or_default();
        let _ = tx.send((stdout_tail, stderr_tail));
    });
    let (stdout_tail, stderr_tail) = rx.recv_timeout(TEE_GRACE).unwrap_or_default();

    if status_signal(&status) == Some(2) {
        return exit_code;
    }

    respond(
        load_result,
        &device,
        &GenerateInput {
            status: exit_code,
            command: &full_command,
            elapsed,
            stdout_tail: &stdout_tail,
            stderr_tail: &stderr_tail,
        },
    );
    exit_code
}

fn respond(
    load_result: anyhow::Result<(Qwen3Model, Tokenizer)>,
    device: &Device,
    input: &GenerateInput<'_>,
) {
    let (model, tokenizer) = match load_result {
        Ok(loaded) => loaded,
        Err(error) => {
            warn(&error);
            return;
        }
    };
    match generate(&model, &tokenizer, device, input) {
        Ok(text) if !text.is_empty() => {
            eprintln!("{text}");
            let _ = io::stderr().flush();
        }
        Ok(_) => {}
        Err(error) => warn(&error),
    }
}

fn join_command(command: &str, args: &[String]) -> String {
    std::iter::once(command)
        .chain(args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

fn spawn_failure_message(command: &str, error: &io::Error) -> String {
    match error.kind() {
        io::ErrorKind::NotFound => format!("{command}: command not found"),
        io::ErrorKind::PermissionDenied => format!("{command}: permission denied"),
        _ => format!("{command}: {error}"),
    }
}

struct GenerateInput<'a> {
    status: i32,
    command: &'a str,
    elapsed: Duration,
    stdout_tail: &'a [u8],
    stderr_tail: &'a [u8],
}

fn generate(
    model: &Qwen3Model,
    tokenizer: &Tokenizer,
    device: &Device,
    input: &GenerateInput<'_>,
) -> anyhow::Result<String> {
    let stdout_ids = tokenizer.encode(&sanitize(input.stdout_tail))?;
    let stderr_ids = tokenizer.encode(&sanitize(input.stderr_tail))?;

    let command_capped: String = input.command.chars().take(200).collect();
    let duration = chat::format_duration(input.elapsed);
    let prompt_ids = assemble_prompt(
        tokenizer,
        input.status,
        &command_capped,
        &duration,
        &stdout_ids,
        &stderr_ids,
    )?;

    let mut cache = KvCache::new();
    let input_tensor = Tensor::<2, Int>::from_data(
        TensorData::new(
            prompt_ids.iter().map(|&id| id as i32).collect::<Vec<i32>>(),
            [1, prompt_ids.len()],
        ),
        device,
    );
    let hidden = model.forward(input_tensor, 0, &mut cache);
    let last = hidden.slice_dim(1, prompt_ids.len() - 1..prompt_ids.len());
    let mut logits = model.lm_head(last);

    let mut rng = rng::Rng::new();
    let mut generated = Vec::new();
    for position in (prompt_ids.len()..).take(GENERATION_CAP) {
        let values = logits.into_data().try_to_vec::<f32>()?;
        let next = rng::sample_top_k(&values, &mut rng, TEMPERATURE, TOP_K);
        if next == tokenizer.im_end_id() {
            break;
        }
        generated.push(next);

        let input = Tensor::<2, Int>::from_data(TensorData::new(vec![next as i32], [1, 1]), device);
        logits = model.lm_head(model.forward(input, position, &mut cache));
    }

    Ok(tokenizer.decode(&generated)?.trim_end().to_owned())
}

fn assemble_prompt(
    tokenizer: &Tokenizer,
    status: i32,
    command: &str,
    duration: &str,
    stdout_ids: &[u32],
    stderr_ids: &[u32],
) -> anyhow::Result<Vec<u32>> {
    // The static fragments only depend on whether either tail is non-empty,
    // so encode each variant once instead of re-tokenizing on every shrink.
    let status_line = chat::status_line(status, command, duration);
    let mut prefix_plain = String::from(chat::USER_START);
    prefix_plain.push_str(&status_line);
    let mut prefix_with_stdout = prefix_plain.clone();
    prefix_with_stdout.push_str(chat::STDOUT_HEADER);

    let prefix_plain_ids = tokenizer.encode(&prefix_plain)?;
    let prefix_with_stdout_ids = tokenizer.encode(&prefix_with_stdout)?;
    let middle_empty_ids = tokenizer.encode("")?;
    let middle_stdout_ids = tokenizer.encode("\n")?;
    let middle_stderr_ids = tokenizer.encode(chat::STDERR_HEADER)?;
    let mut middle_both = String::from("\n");
    middle_both.push_str(chat::STDERR_HEADER);
    let middle_both_ids = tokenizer.encode(&middle_both)?;
    let assistant_ids = tokenizer.encode(chat::ASSISTANT_START)?;

    // Given the current takes, which static fragments are emitted?
    let fragments = |stdout_take: usize, stderr_take: usize| {
        let prefix = if stdout_take > 0 {
            &prefix_with_stdout_ids
        } else {
            &prefix_plain_ids
        };
        let middle = match (stdout_take > 0, stderr_take > 0) {
            (true, true) => &middle_both_ids,
            (true, false) => &middle_stdout_ids,
            (false, true) => &middle_stderr_ids,
            (false, false) => &middle_empty_ids,
        };
        (prefix, middle)
    };
    let prompt_len = |stdout_take: usize, stderr_take: usize| {
        let (prefix, middle) = fragments(stdout_take, stderr_take);
        prefix.len() + stdout_take + middle.len() + stderr_take + assistant_ids.len()
    };

    let mut stdout_take = stdout_ids.len().min(STDOUT_TOKENS);
    let mut stderr_take = stderr_ids
        .len()
        .min(CONTEXT_TOKENS.saturating_sub(stdout_take));

    // Drop the whole overage at once (stderr first, as before). Dropping to
    // zero can also remove a header, so the next length check may shrink
    // further; that converges in at most a couple of iterations.
    let budget = MODEL_CONTEXT - GENERATION_CAP;
    loop {
        let total = prompt_len(stdout_take, stderr_take);
        if total <= budget {
            break;
        }
        if stdout_take == 0 && stderr_take == 0 {
            anyhow::bail!("prompt too long for model context");
        }
        let excess = total - budget;
        if stderr_take > 0 {
            stderr_take = stderr_take.saturating_sub(excess);
        } else {
            stdout_take = stdout_take.saturating_sub(excess);
        }
    }

    let (prefix, middle) = fragments(stdout_take, stderr_take);
    let mut ids = Vec::with_capacity(prompt_len(stdout_take, stderr_take));
    ids.extend_from_slice(prefix);
    ids.extend_from_slice(&stdout_ids[stdout_ids.len() - stdout_take..]);
    ids.extend_from_slice(middle);
    ids.extend_from_slice(&stderr_ids[stderr_ids.len() - stderr_take..]);
    ids.extend_from_slice(&assistant_ids);
    Ok(ids)
}

fn tee<R: Read>(mut reader: R, mut out: impl Write) -> Vec<u8> {
    let mut pending: Vec<u8> = Vec::new();
    let mut tail: Vec<u8> = Vec::new();
    let mut broken = false;
    let mut buffer = [0u8; 8192];

    loop {
        match reader.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if !broken {
                    if out.write_all(&buffer[..n]).is_err() {
                        broken = true;
                    } else {
                        let _ = out.flush();
                    }
                }
                pending.extend_from_slice(&buffer[..n]);
                while let Some(pos) = pending
                    .iter()
                    .position(|&byte| byte == b'\n' || byte == b'\r')
                {
                    let line: Vec<u8> = pending.drain(..=pos).collect();
                    append_tail(&mut tail, &line);
                }
            }
        }
    }
    if !pending.is_empty() {
        append_tail(&mut tail, &pending);
    }
    tail
}

fn append_tail(tail: &mut Vec<u8>, line: &[u8]) {
    if line.contains(&0x1b) {
        return;
    }
    tail.extend_from_slice(line);
    if tail.len() > TAIL_BYTES {
        tail.drain(..tail.len() - TAIL_BYTES);
    }
}

fn sanitize(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).replace("<|", "< |")
}

#[cfg(feature = "embed")]
#[repr(C, align(256))]
struct AlignedBlob<const N: usize>([u8; N]);

#[cfg(feature = "embed")]
const MODEL_LEN: usize = include_bytes!("../model/model.bpk").len();

#[cfg(feature = "embed")]
static MODEL_BLOB: AlignedBlob<MODEL_LEN> = AlignedBlob(*include_bytes!("../model/model.bpk"));
#[cfg(feature = "embed")]
fn load_model(device: &Device) -> anyhow::Result<(Qwen3Model, Tokenizer)> {
    let config = Qwen3Config::parse(include_str!("assets/config.json"))?;
    let model = Qwen3Model::load_bpk_static(config, &MODEL_BLOB.0, device)?;
    let tokenizer = Tokenizer::from_json_str(include_str!("assets/tokenizer.json"))?;
    Ok((model, tokenizer))
}

#[cfg(not(feature = "embed"))]
fn load_model(device: &Device) -> anyhow::Result<(Qwen3Model, Tokenizer)> {
    let config = Qwen3Config::parse(include_str!("assets/config.json"))?;
    let tokenizer = Tokenizer::from_json_str(include_str!("assets/tokenizer.json"))?;
    let candidate = model_path("model.bpk");
    let dir = model_download_dir()
        .ok_or_else(|| anyhow::anyhow!("cannot find home directory for model download"))?;
    let path = fetch::ensure_model_bpk(&candidate, &dir)?;
    let model = Qwen3Model::load_bpk(config, path, device)?;
    Ok((model, tokenizer))
}

fn home_dir() -> Option<std::path::PathBuf> {
    env::var("HOME")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| env::var("USERPROFILE").ok().filter(|value| !value.is_empty()))
        .map(std::path::PathBuf::from)
}

#[cfg(not(feature = "embed"))]
fn model_download_dir() -> Option<std::path::PathBuf> {
    if let Ok(dir) = env::var("MOMMY_MODEL_DIR")
        && !dir.is_empty()
    {
        return Some(std::path::PathBuf::from(dir));
    }
    home_dir().map(|home| home.join(".mommy"))
}

fn model_path(name: &str) -> std::path::PathBuf {
    if let Ok(dir) = env::var("MOMMY_MODEL_DIR")
        && !dir.is_empty()
    {
        return std::path::Path::new(&dir).join(name);
    }
    if let Some(home) = home_dir() {
        let candidate = home.join(".mommy").join(name);
        if candidate.exists() {
            return candidate;
        }
    }
    if let Ok(exe) = env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let candidate = dir.join("model").join(name);
        if candidate.exists() {
            return candidate;
        }
    }
    std::path::Path::new("model").join(name)
}

fn base_command(command: &str, args: &[String]) -> Command {
    let mut cmd = Command::new(command);
    cmd.args(args)
        .env("NO_COLOR", "1")
        .env("CLICOLOR", "0")
        .env_remove("CLICOLOR_FORCE")
        .env_remove("FORCE_COLOR");
    cmd
}

fn status_exit_code(status: &ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    if let Some(signal) = status_signal(status) {
        return 128 + signal;
    }
    1
}

#[cfg(unix)]
fn status_signal(status: &ExitStatus) -> Option<i32> {
    status.signal()
}

#[cfg(not(unix))]
fn status_signal(_status: &ExitStatus) -> Option<i32> {
    None
}

fn spawn_error_code(error: &io::Error) -> i32 {
    match error.kind() {
        io::ErrorKind::NotFound => 127,
        io::ErrorKind::PermissionDenied => 126,
        _ => 1,
    }
}

fn warn(error: &anyhow::Error) {
    eprintln!("mommy: {}", format!("{error:#}").replace('\n', " "));
}
