import numpy as np
import matplotlib.pyplot as plt
import scipy
import copy

def sample_spiral_distribution(size):
    """Sample a spiral distribution.
    This function generates a spiral distribution of points in 2D space.
    
    source: https://github.com/MaxRamgraber/Triangular-Transport-Tutorial/blob/09d58f359823f02ff35315ba6e236ff414c43af7/Figures/Figure%2005/create_figure_05.py#L44
    """
    # First draw some rotation samples from a beta distribution, then scale 
    # them to the range between -pi and +2pi
    seeds = scipy.stats.beta.rvs(
        a       = 4,
        b       = 3,
        size    = size)*3*np.pi-np.pi
    
    # Create a local copy of the rotations
    seeds_orig = copy.copy(seeds)
    
    # Re-normalize the rotations, then scale them to the range between [-3,+3]
    vals    = (seeds+np.pi)/(3*np.pi)*6-3
    
    # Plot the rotation samples on a straight spiral
    X       = np.column_stack((
        np.cos(seeds)[:,np.newaxis],
        np.sin(seeds)[:,np.newaxis]))*((1+seeds+np.pi)/(3*np.pi)*5)[:,np.newaxis]

    # Offset each sample along the spiral's normal vector by scaled Gaussian 
    # noise
    X   += np.column_stack([
        np.cos(seeds_orig),
        np.sin(seeds_orig)])*(scipy.stats.norm.rvs(size=size)*scipy.stats.norm.pdf(vals))[:,np.newaxis]

    return X/2


if __name__ == "__main__":
    # Sample a spiral distribution
    X = sample_spiral_distribution(size=1000)

    # Plot the sampled points
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.axis("equal")
    plt.title("Spiral Distribution")
    plt.show()