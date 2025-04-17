import pymc as pm
from pymc.sampling import jax
import logging

logger = logging.getLogger("pymc")
logger.propagate = False

def bayesian_error_computation(observed_data, measurement_error):
    # Count observed 1s
    k_observed = observed_data.sum()
    N = len(observed_data)
    # Bayesian Model
    with pm.Model() as model:
        # Prior: Assume a Beta distribution for the true proportion
        p = pm.Beta("p", alpha=1, beta=1)  # Non-informative prior

        # e = pm.Normal("e", mu=measurement_error, sigma=0.01)  # Gaussian prior for error rate
        # e = pm.Deterministic("e_clipped", pm.math.clip(e, 0, 0.5))  # Ensure valid range

        # Corrected probability with measurement error
        # p_obs = p * (1 - 2 * e) + e
        # Measurement error model
        p_obs = p * (1 - 2 * measurement_error) + measurement_error

        # Likelihood: Observed data follows a Binomial distribution
        k_likelihood = pm.Binomial("k_obs", n=N, p=p_obs, observed=k_observed)

        # Inference: Run MCMC sampling
        trace = jax.sample_numpyro_nuts(
            10000, tune=2000, progressbar=False, target_accept=0.95
        )
    s = pm.summary(trace, var_names=["p"], hdi_prob=0.95, round_to=5)
    unbiased_score = s.loc["p", "mean"]
    ci_lower = s.loc["p", "hdi_2.5%"]
    ci_upper = s.loc["p", "hdi_97.5%"]
    return unbiased_score, unbiased_score - ci_lower, ci_upper - unbiased_score
