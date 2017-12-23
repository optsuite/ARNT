function M = obliquefactory(n, m)
% A set whose element is n by m matrix with unit 2-norm of each column.

% This is a modified verison of obliquefactory.m in
% Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%	July 16, 2013 (NB) :
%       Added 'transposed' option, mainly for ease of comparison with the
%       elliptope geometry.
%
%	Nov. 29, 2013 (NB) :
%       Added normalize_columns function to make it easier to exploit the
%       bsxfun formulation of column normalization, which avoids using for
%       loops and provides performance gains. The exponential still uses a
%       for loop.
%
%	April 4, 2015 (NB) :
%       Log function modified to avoid NaN's appearing for close by points.
%
%	April 13, 2015 (NB) :
%       Exponential now without for-loops.
%
%   Oct. 8, 2016 (NB)
%       Code for exponential was simplified to only treat the zero vector
%       as a particular case.
%
%  Oct. 21, 2016 (NB)
%       Bug caught in M.log: the function called v = M.proj(x1, x2 - x1),
%       which internally applies transp to inputs and outputs. But since
%       M.log had already taken care of transposing things, this introduced
%       a bug (which only triggered if using M.log in transposed mode.)
%       The code now calls "v = projection(x1, x2 - x1);" since projection
%       assumes the inputs and outputs do not need to be transposed.
%
%   July 20, 2017 (NB)
%       Distance function is now accurate for close-by points. See notes
%       inside the spherefactory file for details. Also improvies distances
%       computation as part of the log function.

% Euclidean inner product
M.inner = @(x, d1, d2) d1(:).'*d2(:);

% Euclidean metric
M.norm = @(x, d) norm(d(:));

% project U to the tangent space T_X M
M.proj = @(X, U) projection(X, U);
    function Up = projection(X, U)
        
        inners = dot(X,U,1);
        Up = U - bsxfun(@times, X, inners);
        
    end

% project the Euclidean gradient to obtain the Riemannian gradient
M.egrad2rgrad = M.proj;

% at point X, given Euclidean gradient and Euclidean Hessian along
% tangent vector H, to obtain the corresponding Riemannian Hessian
% along H
M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, U)
        
        PXehess = projection(X, ehess);
        inners = dot(X, egrad, 1);
        rhess = PXehess - bsxfun(@times, U, inners);
        
    end


M.retr = @retraction;
    % retraction on the oblique manifold
    function y = retraction(x, d, t)
        
        if nargin < 3
            t = 1.0;
        end
        
        y = x + t*d;
        nrms = sqrt(dot(y, y, 1));
        y = bsxfun(@rdivide, y, nrms);
        
    end


M.rand = @() random(n, m);

M.lincomb = @matrixlincomb;

M.zerovec = @(x) zeros(n, m);


    % random point on manifold
    function x = random(n, m)
        
        x = randn(n, m);
        nrms = sqrt(dot(x, x, 1));
        x = bsxfun(@rdivide, x, nrms);
        
    end
end

