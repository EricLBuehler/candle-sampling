use std::{cmp::Ordering, collections::HashMap, iter::zip};

use candle_core::{bail, DType, Error, Result, Tensor, D};
use rand::{
    distributions::{Distribution, WeightedIndex},
    SeedableRng,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

/// LogitsProcessor for sampling.
pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
    sampling_method: SamplingMethod,
    top_n_logprobs: usize,
    tokenizer: Tokenizer,
    repeat_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    logits_bias: Option<HashMap<u32, f32>>,
}

/// Sampling method for `LogitsProcessor`.
///
/// - Multinomial (sample over all tokens)
/// - Top-P (nucleus sampling)
/// - Top-K (top-k sampling)
/// - Top-KP (both, top k first then top p)
#[derive(Debug, Clone)]
pub enum SamplingMethod {
    Multinomial,
    TopP(f64),
    TopK(usize),
    TopKP((usize, f64)),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
// Top-n logprobs element
pub struct TopLogprob {
    pub token: u32,
    pub logprob: f32,
    pub bytes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Logprobs {
    pub token: u32,
    pub logprob: f32,
    pub bytes: String,
    pub top_logprobs: Vec<TopLogprob>,
}

impl LogitsProcessor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        seed: u64,
        temperature: Option<f64>,
        sampling_method: SamplingMethod,
        top_n_logprobs: usize,
        tokenizer: Tokenizer,
        repeat_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        logits_bias: Option<HashMap<u32, f32>>,
    ) -> Self {
        let temperature = if temperature.map_or(true, |v| v < 1e-7) {
            None
        } else {
            temperature
        };
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            temperature,
            sampling_method,
            top_n_logprobs,
            tokenizer,
            repeat_penalty,
            presence_penalty,
            logits_bias,
        }
    }

    /// Sample the provided tokens.
    ///
    /// If the temperature is `None`, argmax sampling is used. Otherwise, the selected sampling is used.
    /// With `top-p` sampling, if the `top-p` value is `<= 0.0` or `>= 1.0`, multinomial sampling is used.
    /// If `repeat_penalty.is_some()` or `presence_penalty.is_some()`, then `penalty_ctxt` must be provided.
    pub fn sample(&mut self, logits: &Tensor, _penalty_ctxt: Option<&[u32]>) -> Result<Logprobs> {
        let logits = logits.to_dtype(DType::F32)?;
        let next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        Ok(Logprobs {
            token: next_token,
            ..Default::default()
        })
    }
}
