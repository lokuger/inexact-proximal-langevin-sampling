# Proximal Langevin Sampling With Inexact Proximal Mapping

This repository implements a Langevin sampling algorithm that includes the evaluation of proximal mappings due to a forward-backward discretization of the energy potential. In cases where this proximal mapping has no exact closed form, we can only iteratively approximate it, making some error in each iteration. In the paper [1] we quantify the bias of the corresponding inexact proximal Langevin sampling due to the errors and show convergence if the errors go to zero over time.
