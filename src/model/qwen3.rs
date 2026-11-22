use std::path::Path;

use burn::module::Module;
use burn::nn::{EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::store::{
    BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use burn::tensor::activation::{silu, softmax};
use serde::Deserialize;

const MASK_NEG: f32 = -1.0e9;

#[derive(Debug, Clone)]
pub(crate) struct Qwen3Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    tie_word_embeddings: bool,
}

#[derive(Debug, Deserialize)]
struct RawRopeParameters {
    rope_theta: f64,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<RawRopeParameters>,
    #[serde(default)]
    tie_word_embeddings: bool,
}

impl Qwen3Config {
    pub(crate) fn parse(raw: &str) -> anyhow::Result<Self> {
        let raw: RawConfig = serde_json::from_str(raw)?;

        let rope_theta = raw
            .rope_parameters
            .as_ref()
            .map(|parameters| parameters.rope_theta)
            .or(raw.rope_theta)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "config has neither rope_parameters.rope_theta nor a flat rope_theta"
                )
            })?;

        if !raw
            .num_attention_heads
            .is_multiple_of(raw.num_key_value_heads)
        {
            anyhow::bail!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                raw.num_attention_heads,
                raw.num_key_value_heads,
            );
        }
        if !raw.head_dim.is_multiple_of(2) {
            anyhow::bail!("head_dim ({}) must be even for RoPE", raw.head_dim);
        }

        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            vocab_size: raw.vocab_size,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta,
            tie_word_embeddings: raw.tie_word_embeddings,
        })
    }
}

pub(crate) struct KvCache {
    keys: Vec<Tensor<4>>,
    values: Vec<Tensor<4>>,
}

impl KvCache {
    pub(crate) fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    fn update(&mut self, layer: usize, key: Tensor<4>, value: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let (key, value) = if layer < self.keys.len() {
            (
                Tensor::cat(vec![self.keys[layer].clone(), key], 2),
                Tensor::cat(vec![self.values[layer].clone(), value], 2),
            )
        } else {
            (key, value)
        };

        if layer == self.keys.len() {
            self.keys.push(key.clone());
            self.values.push(value.clone());
        } else {
            self.keys[layer] = key.clone();
            self.values[layer] = value.clone();
        }

        (key, value)
    }
}

fn linear(input: usize, output: usize, device: &Device) -> Linear {
    LinearConfig::new(input, output)
        .with_bias(false)
        .init(device)
}

fn rms_norm(dim: usize, eps: f64, device: &Device) -> RmsNorm {
    RmsNormConfig::new(dim).with_epsilon(eps).init(device)
}

#[derive(Module, Debug)]
struct Qwen3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3Mlp {
    fn new(config: &Qwen3Config, device: &Device) -> Self {
        Self {
            gate_proj: linear(config.hidden_size, config.intermediate_size, device),
            up_proj: linear(config.hidden_size, config.intermediate_size, device),
            down_proj: linear(config.intermediate_size, config.hidden_size, device),
        }
    }

    fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        let gate = silu(self.gate_proj.forward(input.clone()));
        let up = self.up_proj.forward(input);
        self.down_proj.forward(gate * up)
    }
}

#[derive(Module, Debug)]
struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    rope_cos: Tensor<2>,
    #[module(skip)]
    rope_sin: Tensor<2>,
}

impl Qwen3Attention {
    fn new(config: &Qwen3Config, device: &Device) -> Self {
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let (rope_cos, rope_sin) = rope_tables(
            config.max_position_embeddings,
            head_dim,
            config.rope_theta,
            device,
        );

        Self {
            q_proj: linear(config.hidden_size, heads * head_dim, device),
            k_proj: linear(config.hidden_size, kv_heads * head_dim, device),
            v_proj: linear(config.hidden_size, kv_heads * head_dim, device),
            o_proj: linear(heads * head_dim, config.hidden_size, device),
            q_norm: rms_norm(head_dim, config.rms_norm_eps, device),
            k_norm: rms_norm(head_dim, config.rms_norm_eps, device),
            heads,
            kv_heads,
            head_dim,
            rope_cos,
            rope_sin,
        }
    }

    fn forward(
        &self,
        input: Tensor<3>,
        start_pos: usize,
        layer: usize,
        cache: &mut KvCache,
    ) -> Tensor<3> {
        let [batch, seq, _] = input.dims();
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let head_dim = self.head_dim;
        let device = input.device();

        let query = self
            .q_proj
            .forward(input.clone())
            .reshape([batch, seq, heads, head_dim])
            .permute([0, 2, 1, 3]);
        let query = self.q_norm.forward(query);

        let key = self
            .k_proj
            .forward(input.clone())
            .reshape([batch, seq, kv_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let key = self.k_norm.forward(key);

        let value = self
            .v_proj
            .forward(input)
            .reshape([batch, seq, kv_heads, head_dim])
            .permute([0, 2, 1, 3]);

        let half = head_dim / 2;
        let cos = self
            .rope_cos
            .clone()
            .slice_dim(0, start_pos..start_pos + seq)
            .reshape([1, 1, seq, half]);
        let sin = self
            .rope_sin
            .clone()
            .slice_dim(0, start_pos..start_pos + seq)
            .reshape([1, 1, seq, half]);
        let query = apply_rope(query, cos.clone(), sin.clone());
        let key = apply_rope(key, cos, sin);

        let (key, value) = cache.update(layer, key, value);
        let total = key.dims()[2];

        let group = heads / kv_heads;
        let key = repeat_kv_heads(key, group);
        let value = repeat_kv_heads(value, group);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let scores = query.matmul(key.permute([0, 1, 3, 2])) * scale;
        let scores = if seq > 1 {
            scores + causal_mask(seq, total, &device)
        } else {
            scores
        };
        let weights = softmax(scores, 3);
        let attended = weights.matmul(value).permute([0, 2, 1, 3]);

        let joined = attended.reshape([batch, seq, heads * head_dim]);
        self.o_proj.forward(joined)
    }
}

fn repeat_kv_heads(tensor: Tensor<4>, group: usize) -> Tensor<4> {
    if group == 1 {
        return tensor;
    }
    let [batch, kv_heads, total, head_dim] = tensor.dims();
    tensor
        .reshape([batch, kv_heads, 1, total, head_dim])
        .expand([batch, kv_heads, group, total, head_dim])
        .reshape([batch, kv_heads * group, total, head_dim])
}

#[derive(Module, Debug)]
struct Qwen3DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Qwen3Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Qwen3Mlp,
}

impl Qwen3DecoderLayer {
    fn new(config: &Qwen3Config, device: &Device) -> Self {
        Self {
            input_layernorm: rms_norm(config.hidden_size, config.rms_norm_eps, device),
            self_attn: Qwen3Attention::new(config, device),
            post_attention_layernorm: rms_norm(config.hidden_size, config.rms_norm_eps, device),
            mlp: Qwen3Mlp::new(config, device),
        }
    }

    fn forward(
        &self,
        input: Tensor<3>,
        start_pos: usize,
        layer: usize,
        cache: &mut KvCache,
    ) -> Tensor<3> {
        let normalized = self.input_layernorm.forward(input.clone());
        let attended = self.self_attn.forward(normalized, start_pos, layer, cache);
        let summed = input + attended;
        let normalized = self.post_attention_layernorm.forward(summed.clone());
        let mlp = self.mlp.forward(normalized);
        summed + mlp
    }
}

#[derive(Module, Debug)]
pub(crate) struct Qwen3Model {
    embed_tokens: burn::nn::Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    #[module(skip)]
    config: Qwen3Config,
}

impl Qwen3Model {
    fn new(config: Qwen3Config, device: &Device) -> Self {
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(linear(config.hidden_size, config.vocab_size, device))
        };

        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers: (0..config.num_hidden_layers)
                .map(|_| Qwen3DecoderLayer::new(&config, device))
                .collect(),
            norm: rms_norm(config.hidden_size, config.rms_norm_eps, device),
            lm_head,
            config,
        }
    }

    pub(crate) fn load(
        config: Qwen3Config,
        safetensors_path: impl AsRef<Path>,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut model = Self::new(config, device);
        let mut store = SafetensorsStore::from_file(safetensors_path.as_ref())
            .with_from_adapter(PyTorchToBurnAdapter)
            .with_key_remapping(r"^model\.", "")
            .allow_partial(true);
        let result = model
            .load_from(&mut store)
            .map_err(|error| anyhow::anyhow!("failed to load weights: {error}"))?;

        check_apply_result(&result, "safetensors file")?;
        Ok(model)
    }

    #[cfg_attr(feature = "embed", allow(dead_code))]
    pub(crate) fn load_bpk(
        config: Qwen3Config,
        path: impl AsRef<Path>,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut model = Self::new(config, device);
        let mut store =
            BurnpackStore::from_file(path.as_ref()).with_from_adapter(HalfPrecisionAdapter::new());
        let result = model
            .load_from(&mut store)
            .map_err(|error| anyhow::anyhow!("failed to load weights: {error}"))?;

        check_apply_result(&result, "burnpack file")?;
        Ok(model)
    }

    #[cfg_attr(not(feature = "embed"), allow(dead_code))]
    pub(crate) fn load_bpk_static(
        config: Qwen3Config,
        bytes: &'static [u8],
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut model = Self::new(config, device);
        let mut store =
            BurnpackStore::from_static(bytes).with_from_adapter(HalfPrecisionAdapter::new());
        let result = model
            .load_from(&mut store)
            .map_err(|error| anyhow::anyhow!("failed to load weights: {error}"))?;

        check_apply_result(&result, "embedded burnpack")?;
        Ok(model)
    }

    pub(crate) fn save_bpk(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        self.save_into(
            &mut BurnpackStore::from_file(path.as_ref())
                .overwrite(true)
                .with_to_adapter(HalfPrecisionAdapter::new()),
        )
        .map_err(|error| anyhow::anyhow!("failed to save weights: {error}"))
    }

    pub(crate) fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        start_pos: usize,
        cache: &mut KvCache,
    ) -> Tensor<3> {
        let mut hidden = self.embed_tokens.forward(input_ids);
        for (layer, decoder) in self.layers.iter().enumerate() {
            hidden = decoder.forward(hidden, start_pos, layer, cache);
        }
        self.norm.forward(hidden)
    }

    pub(crate) fn lm_head(&self, hidden: Tensor<3>) -> Tensor<3> {
        match &self.lm_head {
            Some(head) => head.forward(hidden),
            None => {
                let [batch, seq, hidden_size] = hidden.dims();
                let weight = self.embed_tokens.weight.val().transpose();
                hidden
                    .reshape([batch * seq, hidden_size])
                    .matmul(weight)
                    .reshape([batch, seq, self.config.vocab_size])
            }
        }
    }
}

fn check_apply_result(result: &burn::store::ApplyResult, source: &str) -> anyhow::Result<()> {
    if !result.errors.is_empty() {
        anyhow::bail!("weight shape/dtype errors: {:?}", result.errors);
    }
    if !result.missing.is_empty() {
        let missing: Vec<&str> = result
            .missing
            .iter()
            .map(|(path, _)| path.as_str())
            .collect();
        anyhow::bail!("missing tensors in {source}: {}", missing.join(", "));
    }
    if !result.unused.is_empty() {
        anyhow::bail!("unused tensors in {source}: {}", result.unused.join(", "));
    }
    Ok(())
}

fn rope_tables(
    max_positions: usize,
    head_dim: usize,
    theta: f64,
    device: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|pair| theta.powf(-2.0 * pair as f64 / head_dim as f64) as f32)
        .collect();

    let mut cos_values = Vec::with_capacity(max_positions * half);
    let mut sin_values = Vec::with_capacity(max_positions * half);
    for position in 0..max_positions {
        for &freq in &inv_freq {
            let angle = position as f32 * freq;
            cos_values.push(angle.cos());
            sin_values.push(angle.sin());
        }
    }

    let shape = [max_positions, half];
    (
        Tensor::<2>::from_data(TensorData::new(cos_values, shape), device),
        Tensor::<2>::from_data(TensorData::new(sin_values, shape), device),
    )
}

fn apply_rope(x: Tensor<4>, cos: Tensor<4>, sin: Tensor<4>) -> Tensor<4> {
    let half = x.dims()[3] / 2;
    let x1 = x.clone().slice_dim(3, 0..half);
    let x2 = x.slice_dim(3, half..half * 2);
    let left = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let right = x2 * cos + x1 * sin;
    Tensor::cat(vec![left, right], 3)
}

fn causal_mask(seq: usize, total: usize, device: &Device) -> Tensor<4> {
    let mut values = Vec::with_capacity(seq * total);
    for row in 0..seq {
        for col in 0..total {
            values.push(if col > row + (total - seq) {
                MASK_NEG
            } else {
                0.0
            });
        }
    }
    Tensor::<4>::from_data(TensorData::new(values, [1, 1, seq, total]), device)
}
