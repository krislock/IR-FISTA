# IR-FISTA

This repository contains the code developed for the numerical tests in the following paper.

> Yunier Bello-Cruz, Max L. N. Gonçalves, and Nathan Krislock. On FISTA with a relative error rule. Computational Optimization and Applications, 2022. https://doi.org/10.1007/s10589-022-00421-8

The `src` directory contains our implementation of I-FISTA and IA-FISTA for solving the $H$-weighted nearest correlation matrix (NCM) problem.

To run the numerical tests:

`cd src` <br>
`julia --project=.. runtests.jl`

Uses the MATLAB code [`CorNewton3.m`](https://www.polyu.edu.hk/ama/profile/dfsun/CorNewton3.m) by Houduo Qi, Defeng Sun, and Yan Gao to obtain a good initial point by solving the nearest correlation problem. `CorNewton3.m` is based on the algorithm in the following paper.

> Houduo Qi and Defeng Sun. A quadratically convergent Newton method for computing the nearest correlation matrix. SIAM J. Matrix Anal. Appl., 28(2):360–385, 2006. [doi:10.1137/050624509](https://doi.org/10.1137/050624509)
