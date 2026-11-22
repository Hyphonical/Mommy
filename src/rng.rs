use std::time::SystemTime;

pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        let pid = std::process::id() as u64;

        let mut state = nanos ^ pid.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // Mix once so close seeds (same nanosecond bucket) still diverge.
        state = splitmix64(state);
        if state == 0 {
            state = 0x853C_49E6_748F_EA9B;
        }
        Self { state }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix64(self.state)
    }

    fn next_f64(&mut self) -> f64 {
        // Top 53 bits -> [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub(crate) fn sample_top_k(logits: &[f32], rng: &mut Rng, temperature: f32, top_k: usize) -> u32 {
    let top_k = top_k.min(logits.len());
    if top_k == 0 {
        return 0;
    }

    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(top_k - 1, |a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(top_k);

    let max = indices
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = indices
        .iter()
        .map(|&i| ((logits[i] - max) / temperature).exp() as f64)
        .collect();
    let total: f64 = weights.iter().sum();
    if !matches!(total.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        return indices[0] as u32;
    }

    let mut roll = rng.next_f64() * total;
    for (&index, &weight) in indices.iter().zip(&weights) {
        roll -= weight;
        if roll <= 0.0 {
            return index as u32;
        }
    }
    indices.last().copied().unwrap_or(0) as u32
}
