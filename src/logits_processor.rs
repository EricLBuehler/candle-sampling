use std::{cmp::Ordering, collections::HashMap, iter::zip};

use candle_core::{bail, DType, Error, Result, Tensor};
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    fn apply_logit_bias(&self, probs: &mut [f32]) -> Result<()> {
        if let Some(ref bias) = self.logits_bias {
            for (id, bias_v) in bias {
                let idx = probs.get_mut(*id as usize);
                if let Some(idx) = idx {
                    *idx += bias_v;
                } else {
                    candle_core::bail!(
                        "Token ID `{id}` out of range for probs of length `{}`.",
                        probs.len()
                    );
                }
            }
        }
        Ok(())
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<Logprobs> {
        let mut logits_v: Vec<f32> = logits.to_vec1()?;

        self.apply_logit_bias(&mut logits_v)?;

        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i)
            .unwrap();
        let logprob = logits_v[next_token].log(10.0);
        let tok = logits_v[next_token];

        let mut sorted = logits_v.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Where the next token is in the sorted
        let next_token_index = sorted
            .binary_search_by(|w| {
                if *w <= tok {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_err();
        // These are where the top n are
        let top_n_toks_range =
            next_token_index.saturating_sub(self.top_n_logprobs)..next_token_index;
        // The top n's values
        let top_n_logprobs = sorted[top_n_toks_range]
            .iter()
            .map(|x| x.log(10.0))
            .collect::<Vec<_>>();
        // Find where they actually are in the logits
        let mut top_n_toks = Vec::new();
        for val in top_n_logprobs.iter() {
            let idx = logits_v
                .binary_search_by(|w| {
                    if *w <= *val {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                })
                .unwrap_err();
            top_n_toks.push(idx as u32);
        }

        let mut bytes = Vec::new();
        for tok in &top_n_toks {
            bytes.push(
                self.tokenizer
                    .decode(&[*tok], true)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            );
        }
        let top_logprobs = zip(bytes, zip(top_n_toks, top_n_logprobs))
            .map(|(bytes, (token, logprob))| TopLogprob {
                token,
                logprob,
                bytes,
            })
            .collect::<Vec<_>>();

        Ok(Logprobs {
            token: next_token as u32,
            logprob,
            top_logprobs,
            bytes: self
                .tokenizer
                .decode(&[next_token.try_into().unwrap()], true)
                .map_err(|x| Error::Msg(x.to_string()))?,
        })
    }

    fn sample_multinomial(&mut self, probs: &mut Vec<f32>) -> Result<Logprobs> {
        self.apply_logit_bias(probs)?;

        let distr = WeightedIndex::new(&*probs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng); // "Find the first item which has a weight *higher* than the chosen weight."
        let logprob = probs[next_token].log(10.0);
        let tok = probs[next_token];

        let mut sorted = probs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Where the next token is in the sorted
        let next_token_index = sorted
            .binary_search_by(|w| {
                if *w <= tok {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_err();
        // These are where the top n are
        let top_n_toks_range =
            next_token_index.saturating_sub(self.top_n_logprobs)..next_token_index;
        // The top n's values
        let top_n_logprobs = sorted[top_n_toks_range]
            .iter()
            .map(|x| x.log(10.0))
            .collect::<Vec<_>>();
        // Find where they actually are in the logits
        let mut top_n_toks = Vec::new();
        for val in top_n_logprobs.iter() {
            let idx = probs
                .binary_search_by(|w| {
                    if *w <= *val {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                })
                .unwrap_err();
            top_n_toks.push(idx as u32);
        }

        let mut bytes = Vec::new();
        for tok in &top_n_toks {
            bytes.push(
                self.tokenizer
                    .decode(&[*tok], true)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            );
        }
        let top_logprobs = zip(bytes, zip(top_n_toks, top_n_logprobs))
            .map(|(bytes, (token, logprob))| TopLogprob {
                token,
                logprob,
                bytes,
            })
            .collect::<Vec<_>>();

        Ok(Logprobs {
            token: next_token as u32,
            logprob,
            top_logprobs,
            bytes: self
                .tokenizer
                .decode(&[next_token.try_into().unwrap()], true)
                .map_err(|x| Error::Msg(x.to_string()))?,
        })
    }

    fn sample_topp(&mut self, probs: &mut Vec<f32>, top_p: f32) -> Result<Logprobs> {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        let mut argsort_indices = (0..probs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                probs[*index] = 0.0;
            } else {
                cumsum += probs[*index];
            }
        }

        // Sample with clamped probabilities.
        self.sample_multinomial(probs)
    }

    fn sample_topk(&mut self, probs: &mut Vec<f32>, top_k: usize) -> Result<Logprobs> {
        // Sort probs into descending order (highest probs first)
        probs.sort_by(|x, y| x.total_cmp(y));

        // Clamp smaller probabilities to zero.
        for (index, val) in probs.iter_mut().enumerate() {
            if index >= top_k {
                *val = 0.0;
            }
        }

        // Sample with clamped probabilities.
        self.sample_multinomial(probs)
    }

    fn sample_topkp(&mut self, probs: &mut Vec<f32>, top_k: usize, top_p: f32) -> Result<Logprobs> {
        // Sort probs into descending order (highest probs first)
        probs.sort_by(|x, y| x.total_cmp(y));

        // TOP K
        // Clamp smaller probabilities to zero.
        for (index, val) in probs.iter_mut().enumerate() {
            if index >= top_k {
                *val = 0.0;
            }
        }

        // TOP P
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        let mut argsort_indices = (0..probs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                probs[*index] = 0.0;
            } else {
                cumsum += probs[*index];
            }
        }

        // Sample with clamped probabilities.
        self.sample_multinomial(probs)
    }

    fn apply_repeat_presence_penalty(
        logits: &Tensor,
        presence_penalty: f32,
        repeat_penalty: f32,
        context: &[u32],
    ) -> Result<Tensor> {
        //mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
        let device = logits.device();
        let mut logits = logits.to_vec1::<f32>()?;
        for (token_id, logit) in logits.iter_mut().enumerate() {
            let count = context.iter().filter(|x| **x as usize == token_id).count();
            *logit = *logit
                - count as f32 * repeat_penalty
                - if count > 0 { 1. } else { 0. } * presence_penalty;
        }
        let logits_len = logits.len();
        Tensor::from_vec(logits, logits_len, device)
    }

    /// Sample the provided tokens.
    ///
    /// If the temperature is `None`, argmax sampling is used. Otherwise, the selected sampling is used.
    /// With `top-p` sampling, if the `top-p` value is `<= 0.0` or `>= 1.0`, multinomial sampling is used.
    /// If `repeat_penalty.is_some()` or `presence_penalty.is_some()`, then `penalty_ctxt` must be provided.
    pub fn sample(&mut self, logits: &Tensor, penalty_ctxt: Option<&[u32]>) -> Result<Logprobs> {
        let logits = logits.to_dtype(DType::F32)?;

        let logits = if self.repeat_penalty.is_none() && self.presence_penalty.is_none() {
            logits
        } else {
            if penalty_ctxt.is_none() {
                bail!("Must specify penalty context.");
            }
            Self::apply_repeat_presence_penalty(
                &logits,
                self.presence_penalty.unwrap_or(0.),
                self.repeat_penalty.unwrap_or(0.),
                penalty_ctxt.unwrap(),
            )?
        };

        let next_token = match self.temperature {
            None => self.sample_argmax(logits)?,
            Some(temperature) => {
                dbg!(temperature);
                let logits = (&logits / temperature)?;
                let probs = candle_nn::ops::softmax_last_dim(&logits)?;
                let mut probs: Vec<f32> = probs.to_vec1()?;
                match self.sampling_method {
                    SamplingMethod::Multinomial => self.sample_multinomial(&mut probs)?,
                    SamplingMethod::TopP(top_p) => {
                        if top_p <= 0.0 || top_p >= 1.0 {
                            // simply sample from the predicted probability distribution
                            self.sample_multinomial(&mut probs)?
                        } else {
                            // top-p (nucleus) sampling, clamping the least likely tokens to zero
                            self.sample_topp(&mut probs, top_p as f32)?
                        }
                    }
                    SamplingMethod::TopK(top_k) => self.sample_topk(&mut probs, top_k)?,
                    SamplingMethod::TopKP((top_k, top_p)) => {
                        self.sample_topkp(&mut probs, top_k, top_p as f32)?
                    }
                }
            }
        };
        Ok(next_token)
    }
}
