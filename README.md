# inexact proximal Langevin sampling

This repository implements the proximal Langevin algorithm, an algorithm that samples from log-concave distributions with partly non-smooth potentials. In particular, the proximal operator of the non-smooth part of the potential is allowed to be evaluated only inexactly up to some accuracy $\epsilon$. We ran tests using different distributions and different values of $\epsilon$ to verify the theoretical convergence guarantees that we proved.
