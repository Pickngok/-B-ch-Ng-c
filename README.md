# Informer + Gaussian Output for Time Series

This project demonstrates how to use a modified Informer architecture to predict the distribution (mean and variance) of the next value in a time series.

## Files

- `simulate_data.py`: Generates synthetic time series data from a Hidden Markov Model.
- `model.py`: Defines the `InformerGaussian` model using PyTorch.
- `train.py`: Training script using Gaussian NLL loss.
- `informer_gaussian.ipynb`: Jupyter notebook demonstrating the full experiment.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## References

- [Informer (Zhou et al., 2021)](https://arxiv.org/abs/2012.07436)
- [Mixture Density Networks (Bishop, 1994)](https://publications.aston.ac.uk/id/eprint/373/)