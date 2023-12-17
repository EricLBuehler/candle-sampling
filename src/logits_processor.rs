use std::iter::zip;

use candle_core::{DType, Error, Result, Tensor};
use rand::{distributions::Distribution, SeedableRng};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

/// LogitsProcessor for sampling.
pub struct LogitsProcessor<'a> {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
    sampling_method: SamplingMethod,
    top_n_logprobs: usize,
    tokenizer: &'a Tokenizer,
}

/// Sampling method for `LogitsProcessor`.
///
/// - Multinomial (sample over all tokens)
/// - Top-P (nucleus sampling)
/// - Top-K (top-k sampling)
#[derive(Debug, Clone)]
pub enum SamplingMethod {
    Multinomial,
    TopP(f64),
    TopK(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
// Top-n logprobs element
pub struct TopLogprob {
    pub token: usize,
    pub logprob: f32,
    pub bytes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprobs {
    pub token: usize,
    pub logprob: f32,
    pub bytes: String,
    pub top_logprobs: Vec<TopLogprob>,
}

impl<'a> LogitsProcessor<'a> {
    pub fn new(
        seed: u64,
        temperature: Option<f64>,
        sampling_method: SamplingMethod,
        top_n_logprobs: usize,
        tokenizer: &'a Tokenizer,
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
        }
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<Logprobs> {
        let logits_v: Vec<f32> = logits.to_vec1()?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i)
            .unwrap();
        let logprob = logits_v[next_token].log(10.0);

        let mut sorted = logits_v.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let top_n_toks_range = next_token + 1
            ..if next_token + 1 + self.top_n_logprobs <= logits_v.len() {
                next_token + 1 + self.top_n_logprobs
            } else {
                logits_v.len()
            };
        let top_n_toks = top_n_toks_range.clone().collect::<Vec<_>>();
        let top_n_logprobs = sorted[top_n_toks_range]
            .iter()
            .map(|x| x.log(10.0))
            .collect::<Vec<_>>();
        let mut bytes = Vec::new();
        for tok in &top_n_toks {
            bytes.push(
                self.tokenizer
                    .decode(&[(*tok).try_into().unwrap()], true)
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
            token: next_token,
            logprob,
            top_logprobs,
            bytes: self
                .tokenizer
                .decode(&[next_token.try_into().unwrap()], true)
                .map_err(|x| Error::Msg(x.to_string()))?,
        })
    }

    fn sample_multinomial(&mut self, probs: &Vec<f32>) -> Result<Logprobs> {
        let distr = rand::distributions::WeightedIndex::new(probs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng);
        let logprob = probs[next_token].log(10.0);

        let mut sorted = probs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let top_n_toks_range = next_token + 1
            ..if next_token + 1 + self.top_n_logprobs <= probs.len() {
                next_token + 1 + self.top_n_logprobs
            } else {
                probs.len()
            };
        let top_n_toks = top_n_toks_range.clone().collect::<Vec<_>>();
        let top_n_logprobs = sorted[top_n_toks_range]
            .iter()
            .map(|x| x.log(10.0))
            .collect::<Vec<_>>();
        let mut bytes = Vec::new();
        for tok in &top_n_toks {
            bytes.push(
                self.tokenizer
                    .decode(&[(*tok).try_into().unwrap()], true)
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
            token: next_token,
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
        probs.reverse();

        // Clamp smaller probabilities to zero.
        for (index, val) in probs.iter_mut().enumerate() {
            if index >= top_k {
                *val = 0.0;
            }
        }

        // Sample with clamped probabilities.
        self.sample_multinomial(probs)
    }

    /// Sample the provided tokens.
    ///
    /// If the temperature is `None`, argmax sampling is used. Otherwise, the selected sampling is used.
    /// With `top-p` sampling, if the `top-p` value is `<= 0.0` or `>= 1.0`, multinomial sampling is used.
    pub fn sample(&mut self, logits: &Tensor) -> Result<Logprobs> {
        let logits = logits.to_dtype(DType::F32)?;
        let next_token = match self.temperature {
            None => self.sample_argmax(logits)?,
            Some(temperature) => {
                let logits = (&logits / temperature)?;
                let probs = candle_nn::ops::softmax_last_dim(&logits)?;
                let mut probs: Vec<f32> = probs.to_vec1()?;
                match self.sampling_method {
                    SamplingMethod::Multinomial => self.sample_multinomial(&probs)?,
                    SamplingMethod::TopP(top_p) => {
                        if top_p <= 0.0 || top_p >= 1.0 {
                            // simply sample from the predicted probability distribution
                            self.sample_multinomial(&probs)?
                        } else {
                            // top-p (nucleus) sampling, clamping the least likely tokens to zero
                            self.sample_topp(&mut probs, top_p as f32)?
                        }
                    }
                    SamplingMethod::TopK(top_k) => self.sample_topk(&mut probs, top_k)?,
                }
            }
        };
        Ok(next_token)
    }
}
