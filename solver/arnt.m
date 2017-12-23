function [x, G, out] = arnt(x, fun, M, opts, varargin)
% adaptively regularized Newton Method for optimization on manifold
%   min F(x)   s.t. x in M
% 
% Input: 
%         x --- initial guess
%       fun --- objective function and its gradient
%         M --- manifold
%      opts --- options structure with fields
%               record = 0, no print out
%               maxit  max number of iterations
%               xtol   stop control for ||X_k - X_{k-1}||
%               gtol   stop control for the projected gradient
%               ftol   stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%               gamma1, gamma2, gamma3 and gamma4 parameter for adjusting
%               the regularization parameter
%               tau    initial value of regularization parameter
%               solver_init solver for obtaining a good intial guess
%               opts_init options structure for initial solver with fields
%                        record = 0, no print out
%                        maxit  max number of iterations
%                        xtol   stop control for ||X_k - X_{k-1}||
%                        gtol   stop control for the projected gradient
%                        ftol   stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%               solver_sub  solver for subproblem
%               opts_sub    options structure for subproblem solver with fields
%                        record = 0, no print out
%                        maxit  max number of iterations
%                        xtol   stop control for ||X_k - X_{k-1}||
%                        gtol   stop control for the projected gradient or
%                               the accuracy for solving the Newton direction
%                        ftol   stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                               usually, max{xtol, gtol} > ftol
%                        hess   Euclidean Hessian
%
% Output: 
%         x --- solution
%         G --- gradient at x
%       out --- output information
% -----------------------------------------------------------------------
% Reference: 
%  J. Hu, A. Milzark, Z. Wen and Y. Yuan
%  Adaptive Regularized Newton Method for Riemannian Optimization
%
% Author: J. Hu, Z. Wen
%  Version 1.0 .... 2017/8


%------------------------------------------------------------------------

if nargin < 3
    error('at least three inputs: [x, G, out] = arnt(x, fun, M, opts)');
elseif nargin < 4
    opts = [];
end

%-------------------------------------------------------------------------
% options for the trust region solver
if ~isfield(opts, 'gtol');           opts.gtol = 1e-6;  end % 1e-5
if ~isfield(opts, 'xtol');           opts.xtol = 1e-9;  end
if ~isfield(opts, 'ftol');           opts.ftol = 1e-16; end % 1e-13

if ~isfield(opts, 'eta1');           opts.eta1 = 1e-2;  end
if ~isfield(opts, 'eta2');           opts.eta2 = 0.9;   end
if ~isfield(opts, 'gamma1');         opts.gamma1 = 0.2; end
if ~isfield(opts, 'gamma2');         opts.gamma2 = 1;   end
if ~isfield(opts, 'gamma3');         opts.gamma3 = 1e1;  end
if ~isfield(opts, 'gamma4');         opts.gamma4 = 1e2;  end

if ~isfield(opts, 'maxit');          opts.maxit = 200;  end
if ~isfield(opts, 'record');         opts.record = 0;   end
if ~isfield(opts, 'model');          opts.model = 1;    end

if ~isfield(opts, 'eps');            opts.eps = 1e-14;  end
if ~isfield(opts, 'tau');            opts.tau = 10;     end
if ~isfield(opts, 'kappa');          opts.kappa = 0.1;  end
if ~isfield(opts, 'usenumstab');     opts.usenumstab = 1;  end

if ~isfield(opts, 'solver_sub');  opts.solver_sub = @RGBB;   end

hasRecordFile = 0;
if isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'a+'); hasRecordFile = 1;
end


%--------------------------------------------------------------------------
% copy parameters
xtol    = opts.xtol;    gtol   = opts.gtol;    ftol   = opts.ftol;
eta1    = opts.eta1;    eta2   = opts.eta2;    usenumstab = opts.usenumstab;
gamma1  = opts.gamma1;  gamma2 = opts.gamma2;  gamma3 = opts.gamma3;
gamma4  = opts.gamma4;  maxit = opts.maxit;   record = opts.record; 
eps   = opts.eps;       tau = opts.tau;    kappa  = opts.kappa;

solver_sub  = opts.solver_sub;
fun_extra = opts.fun_extra;

%---- If no initial point x is given by the user, generate one at random.
if ~exist('x', 'var') || isempty(x)
    x = M.rand();
end


%--------------------------------------------------------------------------
% GBB for Good init-data
opts_init = opts.opts_init;
% opts_init = opts.opts_init;
solver_init = opts.solver_init;
% -------------------------------------------------------------------------

% out.nfe = 1;
timetic = tic();
% ------------
% Initialize solution and companion measures: f(x), fgrad(x)

if ~isempty(solver_init)
    if strcmp(func2str(solver_init), 'RGBB') || strcmp(func2str(solver_init), 'RAdaGBB')
        fprintf('initial solver\n');
        t1 = tic; [x,~, outin] = feval(solver_init, x, fun, M, opts_init); t2 = toc(t1);
    else
        t1 = tic; [x,~, outin] = feval(solver_init, x, fun, opts_init); t2 = toc(t1);
    end
    
    init = outin.iter;
    out.nfe = outin.nfe + 1;
    out.x0 = x;
    out.intime = t2;
else
    out.x0 = x; out.nfe = 1; out.intime = 0;
end

% compute function value and Euclidean gradient
[F,Ge] = feval(fun, x, varargin{:});

% Riemannian gradient
G = M.egrad2rgrad(x, Ge);
nrmG = M.norm(x,G);  

xp = x; Fp = F; Gp = G; Gep = Ge;

% data structure of x
if isstruct(x)
    matX = x.U*x.S*x.V';
    xp.matX = matX;
    matXP = matX;
end
    
%------------------------------------------------------------------
% OptM for subproblems in TR
opts_sub.tau   = opts.opts_sub.tau;
opts_sub.gtol  = opts.opts_sub.gtol;
opts_sub.xtol  = opts.opts_sub.xtol;
opts_sub.ftol  = opts.opts_sub.ftol;
opts_sub.record = opts.opts_sub.record;
opts_sub.recordFile = opts.opts_sub.recordFile;
stagnate_check = 0;

if record
    str1 = '    %6s';
    stra = ['%8s','%13s  ',str1,str1,str1,str1,str1,str1,'\n'];
    str_head = sprintf(stra,...
        'iter', 'F', 'nrmG', 'XDiff', 'FDiff', 'mDiff', 'ratio', 'tau');
    str1 = '  %3.2e';
    str_num = ['(%3d,%3d)','  %14.8e', str1,str1,str1,str1,str1,str1,'\n'];
end

if hasRecordFile
    fprintf(fid,stra, ...
        'iter', 'F', 'nrmG', 'XDiff', 'FDiff', 'mDiff', 'ratio', 'tau');
end

% main loop 
for iter = 1:maxit
  
    % set the no. of maximal iter. and stagnate check 
    if nrmG >= 1
        opts_sub.maxit = opts.opts_sub.maxit(1);
        stagnate_check = max(stagnate_check,10);
    elseif nrmG >= 1e-2
        opts_sub.maxit = opts.opts_sub.maxit(2);
        stagnate_check = max(stagnate_check,20);
    elseif nrmG >= 1e-3
        opts_sub.maxit = opts.opts_sub.maxit(3);
        stagnate_check = max(stagnate_check,50);
    elseif nrmG >= 1e-4
        opts_sub.maxit = opts.opts_sub.maxit(4);
        stagnate_check = max(stagnate_check,80);
    else
        opts_sub.maxit = opts.opts_sub.maxit(5);
        stagnate_check = max(stagnate_check,100);
    end
    opts_sub.stagnate_check = stagnate_check;

    % criterion of PCG 
    opts_sub.gtol = min(0.1*nrmG, 0.1);
    rreg = 10 * max(1, abs(Fp)) * eps; % safeguarding
    if usenumstab
        opts_sub.rreg = rreg;
    else
        opts_sub.rreg = 0;
    end

    % subproblem solving
    
    % store the information of current iteration 
    TrRho = tau; % remark: ...
    data.TrRho = TrRho;
    data.XP = xp;
    data.sigma = tau*nrmG; % regularization parameter
    
    
    % store the unvaried information in inner iteration
    if ~isempty(fun_extra)
        data = fun_extra(data);
        opts_sub.hess = @(U) opts.hess(xp, U, data);
    else
        opts_sub.hess = @(U) opts.hess(xp, U, data);
    end
    
    % solve the subproblem
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    hess = opts_sub.hess; 
    if strcmp(func2str(solver_sub), 'RNewton')
        [x, out_sub]= feval(solver_sub, xp, Gep, Gp, hess, M, opts_sub);
    else
        [x, ~, out_sub]= feval(solver_sub, xp, fun, hess, M, opts_sub);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % store the iter. info. of inner iter.
    out.iter_sub(iter) = out_sub.iter; % inner iteration no.
    out.nfe_sub(iter)  = out_sub.nfe; % ehess-matrix product and retraction no.
    out.flag(iter) = out_sub.flag;
    out.time_sub(iter) = toc(timetic);
    
    %------------------------------------------------------------------
    % compute fucntion value and Riemannian gradient
    [F, Ge] = feval(fun, x, varargin{:});
    G = M.egrad2rgrad(x,Ge);
    
    out.nfe = out.nfe + 1;
    
    % compute the real and predicted reduction
    redf = Fp - F + rreg;
    mdiff = out_sub.fval0 - out_sub.fval + rreg;
    
    % compute the ration
    model_decreased = mdiff > 0;
    if ~model_decreased % if the model didn't decrase, ratio = -1
        if record; fprintf('model did not decrease\n'); end
        ratio = -1;
    else 
        if (abs(redf)<=eps && abs(mdiff)<= eps) || redf == mdiff
            ratio = 1;
        else
            ratio = redf/mdiff;
        end
    end

    % data structure of x
    if isstruct(x)
        matX = x.matX;
        XDiff = norm(matX - matXP,'fro');
    else
        XDiff = norm(x-xp,'inf');
    end
    
    FDiff = abs(redf)/(abs(Fp)+1);
    
    % accept X
    if ratio >= eta1 && model_decreased 
        xp = x;  Fp = F; Gp = G; Gep = Ge;    
        nrmG = M.norm(x, G); opts_sub.usezero = 1;
    else
        opts_sub.usezero = 0; % opts_sub.deta = out_sub.deta;
    end
    out.nrmGvec(iter) = nrmG;
    out.fvec(iter) = F;
    
    % ---- record ----
    if record 
        if iter == 1
            fprintf('switch to ARNT method \n');
            fprintf('%s', str_head);
        end
        fprintf(str_num, ...
            iter, out_sub.iter, Fp, nrmG, XDiff, FDiff, mdiff, ratio, tau);
    end
    
    if hasRecordFile
        fprintf(fid, str_num, ...
            iter, out_sub.iter, Fp, nrmG, XDiff, FDiff, mdiff, ratio, tau);
    end
    
    % ---- termination ----
    if nrmG <= gtol || ( (FDiff <= ftol) && ratio > 0 )
        out.msg = 'optimal';
        if nrmG  < gtol, out.msg = strcat(out.msg,'_g'); end
        if FDiff < ftol, out.msg = strcat(out.msg,'_f'); end
        break;
    end
    
    % update regularization parameter
    if ratio >= eta2
        tau = max(gamma1*tau, 1e-13);
        %         tau = gamma1*tau;
    elseif ratio >= eta1
        tau = gamma2*tau;
    elseif ratio > 0
        tau = gamma3*tau;
    else
        tau = gamma4*tau;  
    end
    
    % if negative curvature was encoutered, future update regularization parameter
    if out_sub.flag == -1
        tau = max(tau, out_sub.tau/nrmG + 1e-4);
    end
    
    
end % end outer loop
timetoc = toc(timetic);

% store the iter. no.
out.XDiff = XDiff;
out.FDiff = FDiff;
out.nrmG = nrmG;
out.fval = F;
out.iter = iter;
out.nfe = out.nfe + sum(out.nfe_sub);
out.avginit = sum(out.iter_sub)/iter;
out.time = timetoc;

end




