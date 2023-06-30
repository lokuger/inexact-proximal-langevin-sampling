# Proximal Langevin Sampling With Inexact Proximal Mapping

This repository implements a Langevin sampling algorithm that includes the evaluation of proximal mappings due to a forward-backward discretization of the energy potential. In cases where this proximal mapping has no exact closed form, we can only iteratively approximate it, making some error in each iteration. In the paper [1] we quantify the bias of the corresponding inexact proximal Langevin sampling due to the errors and show convergence if the errors go to zero over time.

## The algorithm
For a target distribution with density proportional to exp(-F(x)-G(x)), with smooth F and potentially non-smooth G, the sampling algorithm takes the form

$$  X^{k+1} \approx^{\epsilon_k} \prox_{\gamma G}(X^k - \gamma \nabla F(X^k) + \sqrt{2\gamma}\,\xi^k),\quad \xi^k \sim \mathrm{N}(0,I_d). $$

where $X^k$ is the sample at time step $k$ with step size $\gamma$, and $\approx^{\epsilon_k}$ denotes that we only evaluate the proximal mapping only inexactly up to error level $\epsilon_k$.

## Structure of the repository
- The algorithm is implemented in inexact_pgla.py, with some options allowing to save or drop samples, setting burn-in time, or using the exact proximal mapping for comparison.
- The files potentials.py and distributions.py are used to define the target distribution exp(-F(x)-G(x)), the corresponding routines to evaluate gradients, (inexact) proximal points, etc. 
- The numerical experiments on posterior distributions from imaging inverse problems that are described in the paper are generated using the files of the form example_*.py.
- Auxiliary files used to compute MAP estimates, optimal regularization parameters and asymptotically unbiased samples are located in pdhg.py, sapg.py, pxmala.py.


[1] to appear :)