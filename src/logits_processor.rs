use candle_core::{DType, Error, Result, Tensor};
use rand::{distributions::Distribution, SeedableRng};

/// LogitsProcessor for sampling.
pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
    sampling_method: SamplingMethod,
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

impl LogitsProcessor {
    pub fn new(seed: u64, temperature: Option<f64>, sampling_method: SamplingMethod) -> Self {
        let temperature = if temperature.map_or(true, |v| v < 1e-7) {
            None
        } else {
            temperature
        };
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            temperature,
            sampling_method,
        }
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        let logits_v: Vec<f32> = logits.to_vec1()?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .unwrap();
        Ok(next_token)
    }

    fn sample_multinomial(&mut self, probs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(probs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    fn sample_topp(&mut self, probs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
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

    fn sample_topk(&mut self, probs: &mut Vec<f32>, top_k: usize) -> Result<u32> {
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
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
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
