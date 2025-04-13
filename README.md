# MartianMCMC: Bayesian Parameter Estimation for Genomic Mixture Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)

A Python implementation of the Metropolis-Hastings Markov Chain Monte Carlo (MCMC) algorithm for estimating parameters in a genomic mixture model. This project demonstrates Bayesian inference techniques for a hypothetical problem involving CpG islands in Martian DNA.

## Background

This repository contains an implementation of the Metropolis MCMC algorithm for parameter estimation in a mixture model context. The specific application models the distribution of CpG regions in genomic sequences, where:

- λ (lambda) represents the proportion of CpG islands in genomic sequences
- p₁ represents the proportion of CpGs in CpG islands
- p₂ represents the proportion of CpGs in non-CpG regions

The model assumes p₁ > p₂, as CpG islands are, by definition, more CpG-rich than non-island regions.

## Features

- **Data Simulation**: Generates synthetic genomic data from mixture distributions
- **MCMC Implementation**: Full Metropolis-Hastings algorithm with proposal tuning
- **Visualization**: Trace plots for monitoring convergence of the Markov chain
- **Statistical Analysis**: Posterior mean calculation and acceptance ratio monitoring
- **Command-line Interface**: Easy to use with customizable output paths

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/AtticusColwell/MartianMCMC.git
cd MartianMCMC
```

Install dependencies:

```bash
pip install numpy matplotlib
```

## Usage

Run the MCMC algorithm with:

```bash
python metropolis.py output_plot.png
```

The script will:
1. Generate synthetic genomic data
2. Run the Metropolis MCMC algorithm
3. Print parameter estimates and acceptance ratio
4. Save trace plots to the specified output file

## Example Output

```
Acceptance Ratio: 0.337
Lambda Mean: 0.193
p1 Mean: 0.605
p2 Mean: 0.101
```

![Example Trace Plots](example_output.png)

## Mathematical Details

The log posterior probability is calculated as:

```
log(p(θ|y)) ∝ log(p(y|θ)) + log(p(θ))
```

where:

```
p(y|θ) = ∏[λ·p₁ʸ·(1-p₁)ⁿ⁻ʸ + (1-λ)·p₂ʸ·(1-p₂)ⁿ⁻ʸ]
```

The model parameters θ = [λ, p₁, p₂] are constrained by:
- λ ∈ [0,1)
- p₁ ∈ (0,1]
- p₂ ∈ [0,1)
- p₁ > p₂

## Customization

You can modify several aspects of the MCMC implementation:

- **Chain Length**: Change the `nrep` parameter in `run_metropolis_mcmc()`
- **Proposal Distribution**: Adjust the standard deviations in the `proposal()` function
- **Initial Values**: Modify the starting values in `run_metropolis_mcmc()`
- **Burn-in Period**: Change the `burn_in` variable in `plot_results()`

## Metropolis Algorithm Overview

The implementation follows these steps:
1. Initialize parameter values
2. For each iteration:
   - Propose new parameter values using a jumping distribution
   - Calculate the acceptance ratio based on posterior probabilities
   - Accept or reject the proposal based on this ratio
   - Record the current state of the chain
3. Discard burn-in period and analyze the posterior distribution

## R Implementation

The repository also includes the original R implementation for reference and comparison.

## License

This project is licensed under the MIT License - see the LICENSE file for details.