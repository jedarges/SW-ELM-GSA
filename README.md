# SW-ELM
 Sparse weight-ELM code for variance-based global sensitivity analysis
 
 John Darges - jedarges@ncsu.edu

 This repository is a companion to the manuscript: 

 "Extreme learning machines for variance-based global sensitivity analysis"

 https://arxiv.org/abs/2201.05586
 
- The code implements ELM as a surrogate method for Sobol' analysis. The ELM Sobol' indices are computed analytically from the network parameters.

- The "example.m" file gives an example of how to implement ELM to do GSA

- "elm_sobol_inds.m" is the main feature, a MATLAB function that computes Sobol' indices from network parameters
