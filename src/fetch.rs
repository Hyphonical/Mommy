use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const MODEL_URL: &str = "https://huggingface.co/Hyphonical/Mommy/resolve/main/model.bpk";
/// Upper bound while downloading; the model file is ~128 MB.
const DOWNLOAD_LIMIT: u64 = 1024 * 1024 * 1024;

/// Return `existing` when it is already on disk, otherwise download
/// `model.bpk` into `dir` once and reuse it forever after.
pub(crate) fn ensure_model_bpk(existing: &Path, dir: &Path) -> anyhow::Result<PathBuf> {
    if existing.exists() {
        return Ok(existing.to_path_buf());
    }

    let target = dir.join("model.bpk");
    if target.exists() {
        return Ok(target);
    }

    eprintln!("mommy: downloading model (~128 MB) to {}", target.display());

    fs::create_dir_all(dir).map_err(|error| {
        anyhow::anyhow!("failed to create {}: {error}", dir.display())
    })?;

    let tmp = dir.join("model.bpk.part");
    let _ = fs::remove_file(&tmp);

    let mut response = ureq::get(MODEL_URL)
        .call()
        .map_err(|error| anyhow::anyhow!("model download failed: {error}"))?;
    let mut reader = response
        .body_mut()
        .with_config()
        .limit(DOWNLOAD_LIMIT)
        .reader();

    let mut file = fs::File::create(&tmp)
        .map_err(|error| anyhow::anyhow!("failed to create {}: {error}", tmp.display()))?;
    io::copy(&mut reader, &mut file)
        .map_err(|error| anyhow::anyhow!("model download failed: {error}"))?;
    drop(file);
    drop(reader);

    fs::rename(&tmp, &target)
        .map_err(|error| anyhow::anyhow!("failed to save {}: {error}", target.display()))?;

    eprintln!("mommy: saved to {}", target.display());
    Ok(target)
}
