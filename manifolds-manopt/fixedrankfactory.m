function M = fixedrankfactory(m, n, k)
% A set whose element is a m by n matrix with fixed rank k. This is first
% given by Vandereycken in his paper "Low-rank matrix completion by 
% Riemannian optimization, SIAM J. Optim., 23(2), 1214–1236." 
%
% X is represented by a structure, m by k matrix U, n by k matrix V, k by k
% diagonal matrix S, which are the SVD of X, i.e., X = U*S*V'.
%
% Tangent vectors at X is also represented with a similar structure: m by k
% matrix Up, k by k matrix M and n by k matrix Vp such that Up'*U = 0 and
% Vp'*V = 0. Its corresponding matrix representation is 
%    Z = U*M*V' + Up*V' + U*Vp'.
%
% This is a modified verison of fixedrankembeddedrankfactory.m in
% Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%	Feb. 20, 2014 (NB):
%       Added function tangent to work with checkgradient.
%
%   June 24, 2014 (NB):
%       A couple modifications following
%       Bart Vandereycken's feedback:
%       - The checksum (hash) was replaced for a faster alternative: it's a
%         bit less "safe" in that collisions could arise with higher
%         probability, but they're still very unlikely.
%       - The vector transport was changed.
%       The typical distance was also modified, hopefully giving the
%       trustregions method a better initial guess for the trust region
%       radius, but that should be tested for different cost functions too.
%
%    July 11, 2014 (NB):
%       Added ehess2rhess and tangent2ambient, supplied by Bart.
%
%    July 14, 2014 (NB):
%       Added vec, mat and vecmatareisometries so that hessianspectrum now
%       works with this geometry. Implemented the tangent function.
%       Made it clearer in the code and in the documentation in what format
%       ambient vectors may be supplied, and generalized some functions so
%       that they should now work with both accepted formats.
%       It is now clearly stated that for a point X represented as a
%       triplet (U, S, V), the matrix S needs to be diagonal.

    
    % Euclidean inner product
    M.inner = @(x, d1, d2) d1.M(:).'*d2.M(:) + d1.Up(:).'*d2.Up(:) ...
                                             + d1.Vp(:).'*d2.Vp(:);    
    
    % Euclidean metric                                     
    M.norm = @(x, d) sqrt(M.inner(x, d, d));
            
    % The multiplication between tangent vector Z and matrix W.　If Z is
    % a matrix, it is Z*W. If Z is a structure with fields U,S,V such that
    % Z = U*S*V', it is U*(S*(V'*W)).
    function ZW = tangent_multi(Z, W)
        if ~isstruct(Z)
            ZW = Z*W;
        else
            ZW = Z.U*(Z.S*(Z.V'*W));
        end
    end

    % The multiplication between the transponse of tangent vector Z and
    % matrix W. Similar to tangent_multi.
    function ZtW = tangent_multi_transpose(Z, W)
        if ~isstruct(Z)
            ZtW = Z'*W;
        else
            ZtW = Z.V*(Z.S'*(Z.U'*W));
        end
    end
    
    % The projection of a ambient vector Z to the tangent space with 
    % respect to the Riemannian metric.  
    M.proj = @projection;
    function Zproj = projection(X, Z)
            
        ZV = tangent_multi(Z, X.V);
        UtZV = X.U'*ZV;
        ZtU = tangent_multi_transpose(Z, X.U);

        Zproj.M = UtZV;
        Zproj.Up = ZV  - X.U*UtZV;
        Zproj.Vp = ZtU - X.V*UtZV';

    end

    M.egrad2rgrad = @projection;
    
    % Given the Euclidean gradient at X and the Euclidean Hessian at X
    % along H, where egrad and ehess are vectors in the ambient space and H
    % is a tangent vector at X, returns the Riemannian Hessian at X along
    % H, which is a tangent vector.
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, H)
        
        % Euclidean part
        rhess = projection(X, ehess);
        
        % Curvature part
        T = tangent_multi(egrad, H.Vp)/X.S;
        rhess.Up = rhess.Up + (T - X.U*(X.U'*T));
        T = tangent_multi_transpose(egrad, H.Up)/X.S;
        rhess.Vp = rhess.Vp + (T - X.V*(X.V'*T));
        
    end

    % Transforms a tangent vector Z represented as a structure (Up, M, Vp)
    % into a structure with fields (U, S, V) that represents that same
    % tangent vector in the ambient space of mxn matrices, as U*S*V'.
    % This matrix is equal to X.U*Z.M*X.V' + Z.Up*X.V' + X.U*Z.Vp'. The
    % latter is an mxn matrix, which could be too large to build
    % explicitly, and this is why we return a low-rank representation
    % instead. Note that there are no guarantees on U, S and V other than
    % that USV' is the desired matrix. In particular, U and V are not (in
    % general) orthonormal and S is not (in general) diagonal.
    % (In this implementation, S is identity, but this might change.)
    % Transfer the tangent vector (stucture form) to the matrix form.
    M.tangent2ambient = @tangent2ambient;
    function Zambient = tangent2ambient(X, Z)
        
        Zambient = (X.U*Z.M + Z.Up)*X.V' + X.U *Z.Vp';
    end
    
    % retraction, at current point X (with stucture fields U, S, V), along 
    % tangent vector Z (with structure fields Up, M, Vp) with step size
    % t, the corresponding fesible point after retraction is R_X(tZ) = 
    % U_*S_*V_. 
    M.retr = @retraction;
    function Y = retraction(X, Z, t)
        if nargin < 3
            t = 1.0;
        end

        [Qu, Ru] = qr(Z.Up, 0);
        [Qv, Rv] = qr(Z.Vp, 0);
        
        [Ut, St, Vt] = svd([X.S+t*Z.M , t*Rv' ; t*Ru , zeros(k)]);
        
        Y.U = [X.U Qu]*Ut(:, 1:k);
        Y.V = [X.V Qv]*Vt(:, 1:k);
        Y.S = St(1:k, 1:k) + eps*eye(k);
        
    end
        
    M.rand = @random;
    % Factors U and V live on Stiefel manifolds, hence we will reuse
    % their random generator.
    stiefelm = stiefelfactory(m, k);
    stiefeln = stiefelfactory(n, k);
    function X = random()
        X.U = stiefelm.rand();
        X.V = stiefeln.rand();
        X.S = diag(sort(rand(k, 1), 1, 'descend'));
    end
    
    % Generate a random tangent vector at X.
    M.randvec = @randomvec;
    function Z = randomvec(X)
        Z.Up = randn(m, k);
        Z.Vp = randn(n, k);
        Z.M  = randn(k);
        Z = tangent(X, Z);
        nrm = M.norm(X, Z);
        Z.Up = Z.Up / nrm;
        Z.Vp = Z.Vp / nrm;
        Z.M  = Z.M  / nrm;
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) struct('Up', zeros(m, k), 'M', zeros(k, k), ...
                                                        'Vp', zeros(n, k));
    

    M.vec = @vec;
    function Zvec = vec(X, Z)
        Zamb = tangent2ambient(X, Z);
        Zvec = Zamb(:);
    end

end

% Linear combination of tangent vectors
function d = lincomb(x, a1, d1, a2, d2)

    if nargin == 3
        d.Up = a1*d1.Up;
        d.Vp = a1*d1.Vp;
        d.M  = a1*d1.M;
    elseif nargin == 5
        d.Up = a1*d1.Up + a2*d2.Up;
        d.Vp = a1*d1.Vp + a2*d2.Vp;
        d.M  = a1*d1.M  + a2*d2.M;
    else
        error('fixedrank.lincomb takes either 3 or 5 inputs.');
    end

end
