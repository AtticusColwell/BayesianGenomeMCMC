import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def setup_command_line_args():
    parser = argparse.ArgumentParser(description='Run Metropolis MCMC for Martian DNA Problem')
    parser.add_argument('output_file', type=str, help='Path to save the plot')
    return parser.parse_args()

def generate_data():
    # Set the random seed for reproducibility
    np.random.seed(3)
    
    # Generate data similar to the R script
    N = 100
    n = np.random.poisson(50, N)
    y1 = np.random.binomial(n, 0.6)  # CpG
    y2 = np.random.binomial(n, 0.1)  # non-CpG
    y = np.where(np.random.uniform(0, 1, N) < 0.2, y1, y2)
    
    return N, n, y

def log_posterior(theta, n, y):
    """
    Calculate the log posterior probability
    
    Parameters:
    -----------
    theta : array-like
        [lambda, p1, p2] parameters
    n : array-like
        Size parameters for binomial distribution
    y : array-like
        Observations
    
    Returns:
    --------
    float
        Log posterior probability
    """
    lambda_val = theta[0]
    p1 = theta[1]
    p2 = theta[2]
    
    # Check constraints
    if (lambda_val >= 1 or p1 > 1 or p2 >= 1 or 
        lambda_val <= 0 or p1 <= 0 or p2 < 0 or p1 < p2):
        return -999999
    
    # Calculate log posterior
    log_prob_island = y * np.log(p1) + (n - y) * np.log(1 - p1)
    log_prob_non_island = y * np.log(p2) + (n - y) * np.log(1 - p2)
    
    # Use log-sum-exp trick for numerical stability
    max_log_prob = np.maximum(log_prob_island, log_prob_non_island)
    logsumexp = max_log_prob + np.log(
        lambda_val * np.exp(log_prob_island - max_log_prob) + 
        (1 - lambda_val) * np.exp(log_prob_non_island - max_log_prob)
    )
    
    return np.sum(logsumexp)

def proposal(theta):
    """
    Generate proposed parameter values from jumping distribution
    
    Parameters:
    -----------
    theta : array-like
        Current parameter values [lambda, p1, p2]
    
    Returns:
    --------
    array-like
        Proposed parameter values
    """
    # Standard deviations for the jumping distribution
    sds = np.array([0.01, 0.01, 0.01])
    return theta + np.random.normal(0, 1, 3) * sds

def run_metropolis_mcmc(n, y, nrep=3000):
    """
    Run the Metropolis MCMC algorithm
    
    Parameters:
    -----------
    n : array-like
        Size parameters for binomial distribution
    y : array-like
        Observations
    nrep : int
        Number of iterations
    
    Returns:
    --------
    tuple
        (MCMC chain, acceptance ratio)
    """
    # Initialize parameters
    lambda_val = 0.2
    p1 = 0.6
    p2 = 0.1
    
    # Initialize MCMC chain
    mchain = np.zeros((nrep, 3))
    mchain[0] = np.array([lambda_val, p1, p2])
    
    # Initialize acceptance counter
    acc = 0
    
    # Run MCMC
    for i in range(1, nrep):
        # Current parameter values
        theta = mchain[i-1]
        
        # Propose new parameter values
        theta_candidate = proposal(theta)
        
        # Calculate acceptance ratio
        alpha = log_posterior(theta_candidate, n, y) - log_posterior(theta, n, y)
        
        # Accept or reject proposal
        if np.random.uniform(0, 1) <= np.exp(alpha):
            acc += 1
            mchain[i] = theta_candidate
        else:
            mchain[i] = theta
    
    # Calculate acceptance ratio
    accept_ratio = acc / nrep
    
    return mchain, accept_ratio

def plot_results(mchain, accept_ratio, output_file):
    """
    Plot the MCMC chain results
    
    Parameters:
    -----------
    mchain : array-like
        MCMC chain
    accept_ratio : float
        Acceptance ratio
    output_file : str
        Path to save the plot
    """
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define burn-in period
    burn_in = 100
    
    # Plot lambda
    axes[0].plot(mchain[burn_in:, 0])
    axes[0].set_title('lambda')
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel('Iteration')
    
    # Plot p1
    axes[1].plot(mchain[burn_in:, 1])
    axes[1].set_title('p1')
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('Iteration')
    
    # Plot p2
    axes[2].plot(mchain[burn_in:, 2])
    axes[2].set_title('p2')
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel('Iteration')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file)
    
    # Print statistics
    lambda_mean = np.mean(mchain[burn_in:, 0])
    p1_mean = np.mean(mchain[burn_in:, 1])
    p2_mean = np.mean(mchain[burn_in:, 2])
    
    print(f'Acceptance Ratio: {accept_ratio:.3f}')
    print(f'Lambda Mean: {lambda_mean:.3f}')
    print(f'p1 Mean: {p1_mean:.3f}')
    print(f'p2 Mean: {p2_mean:.3f}')

def main():
    # Parse command line arguments
    args = setup_command_line_args()
    
    # Generate data
    N, n, y = generate_data()
    
    # Run MCMC
    mchain, accept_ratio = run_metropolis_mcmc(n, y)
    
    # Plot results
    plot_results(mchain, accept_ratio, args.output_file)

if __name__ == "__main__":
    main()