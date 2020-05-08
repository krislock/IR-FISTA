# IR-FISTA

This repository contains the code developed for the numerical tests in the following paper.

> Yunier Bello-Cruz, Max L. N. Gonçalves, and Nathan Krislock. On Inexact Accelerated Proximal Gradient Methods with Relative Error Rules. Submitted May 6, 2020.
[ Preprint ](http://www.optimization-online.org/DB_HTML/2020/05/7778.html)

The `src` directory contains our implementation of I-FISTA, IE-FISTA, and IA-FISTA for solving the $H$-weighted nearest correlation matrix (NCM) problem.

To run the numerical tests:

`cd src` <br>
`julia runtests.jl`

Uses the MATLAB code [`CorNewton3.m`](https://www.polyu.edu.hk/ama/profile/dfsun/CorNewton3.m) by Houduo Qi, Defeng Sun, and Yan Gao to obtain a good initial point by solving the nearest correlation problem. `CorNewton3.m` is based on the algorithm in the following paper.

> Houduo Qi and Defeng Sun. A quadratically convergent Newton method for computing the nearest correlation matrix. SIAM J. Matrix Anal. Appl., 28(2):360–385, 2006. [doi:10.1137/050624509](https://doi.org/10.1137/050624509)
