import jax.numpy as jnp
from . import cca_core


def compute_pwcca(acts1, acts2, epsilon=0.):
    """ Computes projection weighting for weighting CCA coefficients 
    
    Args:
         acts1: 2d numpy array, shaped (neurons, num_datapoints)
	 acts2: 2d numpy array, shaped (neurons, num_datapoints)

    Returns:
	 Original cca coefficient mean and weighted mean

    """
    sresults = cca_core.get_cca_similarity(acts1, acts2, epsilon=epsilon, 
					   compute_dirns=False, compute_coefs=True, verbose=False)
    
    if jnp.sum(sresults["x_idxs"]) <= jnp.sum(sresults["y_idxs"]):
        dirns = jnp.dot(sresults["coef_x"], 
                    (acts1[sresults["x_idxs"]] - \
                     sresults["neuron_means1"][sresults["x_idxs"]])) + sresults["neuron_means1"][sresults["x_idxs"]]
        coefs = sresults["cca_coef1"]
        acts = acts1
        idxs = sresults["x_idxs"]
    else:
        dirns = jnp.dot(sresults["coef_y"], 
                    (acts1[sresults["y_idxs"]] - \
                     sresults["neuron_means2"][sresults["y_idxs"]])) + sresults["neuron_means2"][sresults["y_idxs"]]
        coefs = sresults["cca_coef2"]
        acts = acts2
        idxs = sresults["y_idxs"]
    P, _ = jnp.linalg.qr(dirns.T)
    weights = jnp.sum(jnp.abs(jnp.dot(P.T, acts[idxs].T)), axis=1)
    weights = weights/jnp.sum(weights)
    
    return jnp.sum(weights*coefs).to_py(), weights.to_py(), coefs.to_py() 
