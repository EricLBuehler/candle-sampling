# candle-sampling
[![Continuous integration](https://github.com/EricLBuehler/candle-sampling/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-sampling/actions/workflows/ci.yml)
[![Documentation](https://github.com/EricLBuehler/candle-sampling/actions/workflows/docs.yml/badge.svg)](https://ericlbuehler.github.io/candle-sampling/candle_sampling/)

Sampling techniques for Candle.

Currently implemented methods are:
- multinomial
- top-k
- top-p
- repeat penalty

Beam search is being developed, but Candle's `Tensor` lacks necessary features for the implementation. See the issue [here](https://github.com/huggingface/candle/issues/1279).