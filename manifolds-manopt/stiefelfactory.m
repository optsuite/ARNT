function M = stiefelfactory(n, p, k)
% A set whose element is a n by p orthogonal matrix. If k is 1, the point 
% is represented by a n by p orthogonal matrix. If k is larger than 1,
% this is a Cartesian product of k Stiefel manifolds. The point is
% represented by a n by p by k tensor with  
%        X(:,:,i)'*X(:,:,i) = eye(p), i = 1, ..., k.
%
% The default value of k is 1.
%  
% This is a modified verison of stiefelfactory.m in
% Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%  July  5, 2013 (NB) : Added ehess2rhess.
%  Jan. 27, 2014 (BM) : Bug in ehess2rhess corrected.
%  June 24, 2014 (NB) : Added true exponential map and changed the randvec
%                       function so that it now returns a globally
%                       normalized vector, not a vector where each
%                       component is normalized (this only matters if k>1).
    
    if ~exist('k', 'var') || isempty(k)
        k = 1;
    end

    % Euclidean inner product
    M.inner = @(x, d1, d2) d1(:).'*d2(:);
    
    % Euclidean norm
    M.norm = @(x, d) norm(d(:));
    
    
    % Project U to the tangent space T_X M
    M.proj = @projection;
    function Up = projection(X, U)
        
        XtU = multiprod(multitransp(X), U);
        symXtU = multisym(XtU);
        Up = U - multiprod(X, symXtU);
        
    end
    
    % project the Euclidean gradient to obtain the Riemannian gradient
	M.egrad2rgrad = M.proj;
    
    
    % at point X, given Euclidean gradient and Euclidean Hessian along
    % tangent vector H, to obtain the corresponding Riemannian Hessian
    % along H
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, H)
        XtG = multiprod(multitransp(X), egrad);
        symXtG = multisym(XtG);
        HsymXtG = multiprod(H, symXtG);
        rhess = projection(X, ehess - HsymXtG);
    end
    

    % retraction
    M.retr = @retraction;
    function Y = retraction(X, U, t)
        if nargin < 3
            t = 1.0;
        end
        Y = X + t*U;
        for i = 1:k
            Y(:,:,i) = Y(:,:,i)*(Y(:,:,i)'*Y(:,:,i))^(-0.5);
        end

    end

    % random feasible point on manifold
    M.rand = @random;
    function X = random()
        X = zeros(n, p, k);
        for i = 1 : k
            [Q, ~] = qr(randn(n, p), 0); 
            X(:, :, i) = Q;
        end
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(n, p, k);
        

end
