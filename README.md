# candle-sampling
[![Continuous integration](https://github.com/EricLBuehler/candle-sampling/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-sampling/actions/workflows/ci.yml)
[![Documentation](https://github.com/EricLBuehler/candle-sampling/actions/workflows/docs.yml/badge.svg)](https://ericlbuehler.github.io/candle-sampling/candle_sampling/)

Sampling techniques for Candle.

Currently implemented methods are:
- multinomial (ancestral)
- top-k
- top-p (nucleus)
- top-n logprobs
- repeat penalty (frequency penalty)
- presence penalty
- logit bias