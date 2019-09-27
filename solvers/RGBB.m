function [x, f, out] = RGBB(x, fun, M, opts, varargin)

% Riemannian gradient method with BB step size
%   min F(x), s.t., x in M
%
% Input:
%           x --- initial guess
%         fun --- objective function and its gradient:
%                 [F, G] = fun(X,  data1, data2)
%                 F, G are the objective function value and gradient, repectively
%                 data1, data2 are addtional data, and can be more
%                 Calling syntax:
%                   [X, out]= OptStiefelGBB(X0, @fun, opts, data1, data2);
%           M --- manifold
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 maxit       max number of iterations
%                 xtol        stop control for ||X_k - X_{k-1}||
%                 gtol        stop control for the projected gradient
%                 ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%                 alpha       initial step size
%        rhols, eta, nt       parameters in line search
%   
% Output:
%           x --- solution
%           f --- function value at x
%         out --- output information
% -----------------------------------------------------------------------
% Reference: 
%  J. Hu, A. Milzark, Z. Wen and Y. Yuan
%  Adaptive Quadratically Regularized Newton Method for Riemannian Optimization
%
% Author: J. Hu, Z. Wen
%  Version 1.0 .... 2017/8

if nargin < 3
    error('at least three inputs: [x, f, out] = RGBB(x, fun, M, opts)');
elseif nargin < 4
    opts = [];
end

% termination rule
if ~isfield(opts, 'gtol');      opts.gtol = 1e-6;  end % 1e-5
if ~isfield(opts, 'xtol');      opts.xtol = 1e-6;  end % 1e-6
if ~isfield(opts, 'ftol');      opts.ftol = 1e-13; end % 1e-13

% parameters for control the linear approximation in line search,
if ~isfield(opts, 'alpha');     opts.alpha  = 1e-3;   end
if ~isfield(opts, 'rhols');     opts.rhols  = 1e-6;   end
if ~isfield(opts, 'eta');       opts.eta  = 0.2;      end
if ~isfield(opts, 'gamma');     opts.gamma  = 0.85;   end
if ~isfield(opts, 'STPEPS');    opts.STPEPS  = 1e-10; end
if ~isfield(opts, 'nt');        opts.nt  = 3;         end % 3
if ~isfield(opts, 'maxit');     opts.maxit  = 200;   end
if ~isfield(opts, 'eps');       opts.eps = 1e-14;     end
if ~isfield(opts, 'record');    opts.record = 0;      end
if ~isfield(opts, 'radius');    opts.radius = 1;      end
if isfield(opts,  'nt');         opts.nt = 5;          end

hasRecordFile = 0;
if isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'w+'); hasRecordFile = 1;
end

% initial guess
if ~exist('x', 'var') || isempty(x)
    x = problem.M.rand();
end

% copy parameters
gtol = opts.gtol;
xtol = opts.xtol;
ftol = opts.ftol;
maxit = opts.maxit;
rhols = opts.rhols;
eta   = opts.eta;
eps   = opts.eps;
gamma = opts.gamma;
record = opts.record;
nt = opts.nt;
alpha = opts.alpha;

% initial function value and gradient
[f,ge] = feval(fun, x, varargin{:});
g = M.egrad2rgrad(x,ge);

% norm
if isstruct(g)
    matG = M.tangent2ambient(x,g);
    nrmG = norm(matG,'fro');
else
    nrmG = norm(g,'fro');
end

if isstruct(x)
    matX = x.U*x.S*x.V';
end

% initial iter. information 
out.nfe = 1; Q = 1; Cval = f; out.fval0 = f;

%% Print iteration header if debug == 1

if hasRecordFile 
    fprintf(fid,'%4s \t %10s \t %10s \t  %10s \t %10s \t %10s \t %10s \t %10s\n', ...
        'Iter', 'f(X)', 'Cval', 'nrmG', 'XDiff', 'FDiff', 'nls', 'alpha');
end

if record == 10; out.fvec = f; end
out.msg = 'exceed max iteration';

if record
    str1 = '    %6s';
    stra = ['%6s','%12s  ','%12s  ',str1,str1,str1,'   %.5s','  %.6s','\n'];
    str_head = sprintf(stra,...
        'iter', 'F','Cval', 'nrmG', 'XDiff', 'FDiff', 'nls', 'alpha');
    str1 = '  %3.2e';
    str_num = ['%4d','  %14.8e', '  %14.8e', str1,str1,str1, '  %d','  %3.2e','\n'];
end

% loop
for iter = 1:maxit
    
    xp = x; gp = g; fp = f; 
    if isstruct(x), matXP = matX; end 
    if isstruct(g), matGP = matG; end
    nls = 1; deriv = rhols*nrmG^2; 
    d = M.lincomb(x, -1, g);
    
    % curvilinear search
    while 1
        
        x = M.retr(xp, d, alpha);
        [f,ge] = feval(fun, x, varargin{:});
        
        out.nfe = out.nfe + 1;
        if f <=  Cval - alpha*deriv || nls >= 5
            break
        end
        alpha = eta*alpha;
        nls = nls+1;
    end
    
    % Riemannian gradient
    g = M.egrad2rgrad(x,ge); 
    
    % norm
    if isstruct(g)
        matG = M.tangent2ambient(x,g);
        nrmG = norm(matG,'fro');
    else
        nrmG = M.norm(x,g);
    end
    
    out.nrmGvec(iter) = nrmG;

    if record == 10; out.fvec = [out.fvec; f]; end
    
    % difference of x
    if isstruct(x)
        matX = x.U*x.S*x.V';
        s = matX - matXP;
    else
        s = x - xp;
    end
    
    XDiff = norm(s,'inf')/alpha; % (relative Xdiff) ~ g
    FDiff = abs(f-fp)/(abs(fp)+1);
    
    % ---- record ----
    if record 
        if iter == 1
            fprintf('\n%s', str_head);
        end
        fprintf(str_num, iter, f, Cval, nrmG, XDiff, FDiff, nls, alpha);
    end
    
    if hasRecordFile 
        fprintf(fid,...
         '%4d\t %14.13e\t %14.13e\t %3.2e\t %3.2e\t %3.2e\t %2d\t %3.2e\n', ...
            iter, f, Cval, nrmG, XDiff, FDiff, nls, alpha);
    end
    
    
    % check stopping
    crit(iter) = FDiff;
    mcrit = mean(crit(iter-min(nt,iter)+1:iter));
    
    % ---- termination ----
    if nrmG < gtol || XDiff < xtol || FDiff < ftol
        %     if nrmG < gtol || XDiff < xtol || mcrit < ftol
        %    if nrmG < gtol
        out.msg = 'converge';
        if nrmG  < gtol, out.msg = strcat(out.msg,'_g'); end
        if XDiff < xtol, out.msg = strcat(out.msg,'_x'); end
        %         if FDiff < ftol, out.msg = strcat(out.msg,'_f'); end
        if mcrit < ftol, out.msg = strcat(out.msg,'_mf'); end
        break;
    end
    
    % difference of gradient
    if isstruct(g)
        y = matG - matGP;
    else
        y = g - gp;
    end
    
    % BB step size
    sy = abs(iprod(s,y));
    if sy > 0
        if mod(iter,2)==0; alpha = norm(s, 'fro')^2/sy;
        else alpha = sy/ norm(y, 'fro')^2; end
        % safeguarding on alpha
        alpha = max(min(alpha, 1e20), 1e-20);
    end
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + f)/Q;
           
end
out.XDiff = XDiff;
out.FDiff = FDiff;
out.mcrit = mcrit;
out.nrmG = nrmG;
out.fval = f;
out.iter = iter;

% Euclidean inner product
function a = iprod(x,y)
  a = real(sum(sum(conj(x).*y)));
end

end
