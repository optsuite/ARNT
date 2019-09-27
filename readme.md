 # ARNT beta 1.1
 The algorithms in arnt.m and RNewton.m are modified to be more stable, especially when a high accuracy is required.

 # ARNT beta 1.0
 A MATLAB software for solving optimization problems on manifold.

 # Problems and solvers
 - The package contains codes for optimization problems on manifold:   
    ​        min  f(x)  s.t. x in M,
    where M is Riemannian manifold.

 - The manifold structure of ARNT is taken from the Manopt package. Further information can be found on the website www.manopt.org. We thank the authors for kindly sharing their codes. Their specific licenses should be considered before modifying and/or redistributing them.

Applications have been solved by these solvers:

- Homogeneous polynomial optimization problems with multiple spherical constraints：
  $$\max \;  \sum_{1\le i\le n_1, 1\le j \le n_2, 1 \le k \le n_3, 1\le l \le n_4} a_{ijkl} x_i y_j z_k w_l \;  s.t., \|x\|_2 = \|y\|_2 = \|z\|_2 = \|w\|_2= 1,$$
  where $A = (a_{ijkl})$ is a fourth-order tensor of size $n\times n \times n\times n$.
- Maxcut SDP: $\min  \mathrm{Tr}(CX), s.t., X_{ii}=1, X \succeq 0$
- SDP: $\min \mathrm{Tr}(CX), s.t., \mathrm{Tr}(X)=1, X \succeq 0 $
- Low-Rank Nearest Correlation  Estimation: $ \min_{ X \succeq 0} \; \frac{1}{2} \| H \odot (X - C) \|_F^2, \; X_{ii} = 1, \; i = 1, \ldots, n, \; \mathrm{rank}(X) \le p.$
- The Bose–Einstein condensates (BEC) problem
- Linear eigenvalue problems: $\min \mathrm{Tr}(X^{\top}AX), s.t., X^{\top}X =I $
- The electronic structure calculation: the Kohn-Sham total energy minimization and the Hartree-Fock total energy minimization
- Quadratic assignment problem
- Harmonic energy minimization

 # References
- [Jiang Hu, Andre Milzarek, Zaiwen Wen, Yaxiang Yuan. Adaptive Quadratically Regularized Newton Method for Riemannian Optimization. SIAM Journal on Matrix Analysis and Applications, Vol. 39, No. 3, pp. 1181–1207](https://epubs.siam.org/doi/10.1137/17M1142478)

- [Jiang Hu, Bo Jiang, Lin Lin, Zaiwen Wen, Yaxiang Yuan. Structured Quasi-Newton Methods for Optimization with Orthogonality Constraints. SIAM Journal on Scientific Computing, Vol. 41, No. 4, pp. A2239-A2269](https://arxiv.org/abs/1809.00452)

- [Zaiwen Wen and Wotao Yin. A feasible method for optimization with orthogonality constraints. Mathematical Programming (2013): 397-434.](https://link.springer.com/article/10.1007/s10107-012-0584-1)

- [Zaiwen Wen, Andre Milzarek, Michael Ulbrich and Hongchao Zhang, Adaptive regularized self-consistent field iteration with exact Hessian for electronic structure calculation. SIAM Journal on Scientific Computing (2013), A1299-A1324.](https://doi.org/10.1137/120894385)

- [Xinming Wu, Zaiwen Wen, and Weizhu Bao. A regularized Newton method for computing ground states of Bose–Einstein condensates. Journal of Scientific Computing (2017): 303-329.](https://link.springer.com/article/10.1007/s10915-017-0412-0)

- X. Zhang, J. Zhu, Z. Wen, A. Zhou, Gradient-type Optimization Methods for Electronic Structure Calculation, SIAM Journal on Scientific Computing, Vol. 36, No. 3 (2014), pp. C265-C289

- R. Lai, Z. Wen, W. Yin, X. Gu, L. Lui, Folding-Free Global Conformal Mapping for Genus-0 Surfaces by Harmonic Energy Minimization, Journal of Scientfic Computing, 58(2014), 705-725
  
- [Nicolas Boumal , Bamdev Mishra, P.-A. Absil and Rodolphe Sepulchre. Manopt, a Matlab Toolbox for Optimization on Manifolds. Journal of Machine Learning Research (2014) 1455-1459](http://jmlr.org/papers/v15/boumal14a.html)




 # The Authors
 We hope that the package is useful for your application.  If you have any bug reports or comments, please feel free to email one of the toolbox authors:

 * Jiang Hu, jianghu at pku.edu.cn
 * Zaiwen Wen, wenzw at pku.edu.cn

 # Installation
 `>> startup`  

 `>> cd example` 

 `>> test_ncm`


 # Copyright
-------------------------------------------------------------------------
   Copyright (C) 2017, Jiang Hu, Zaiwen Wen

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without  even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>

