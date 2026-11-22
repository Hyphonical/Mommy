use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

pub(crate) struct Tokenizer {
    inner: HfTokenizer,
    im_end_id: u32,
}

impl Tokenizer {
    #[allow(dead_code)]
    pub(crate) fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|error| anyhow::anyhow!("failed to load tokenizer: {error}"))?;
        Self::from_inner(inner)
    }

    #[cfg_attr(not(feature = "embed"), allow(dead_code))]
    pub(crate) fn from_json_str(raw: &str) -> anyhow::Result<Self> {
        let inner = HfTokenizer::from_bytes(raw.as_bytes())
            .map_err(|error| anyhow::anyhow!("failed to load tokenizer: {error}"))?;
        Self::from_inner(inner)
    }

    fn from_inner(inner: HfTokenizer) -> anyhow::Result<Self> {
        let im_end_id = inner
            .token_to_id("<|im_end|>")
            .ok_or_else(|| anyhow::anyhow!("tokenizer is missing the <|im_end|> token"))?;
        Ok(Self { inner, im_end_id })
    }

    pub(crate) fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!("failed to encode text: {error}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub(crate) fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|error| anyhow::anyhow!("failed to decode token ids: {error}"))
    }

    pub(crate) fn im_end_id(&self) -> u32 {
        self.im_end_id
    }
}
