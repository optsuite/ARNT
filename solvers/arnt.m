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
%
% Output:
%         x --- solution
%         G --- gradient at x
%       out --- output information
% -----------------------------------------------------------------------
% Reference:
%  J. Hu, A. Milzark, Z. Wen and Y. Yuan
%  Adaptive Quadratically Regularized Newton Method for Riemannian Optimization
%
% Author: J. Hu, Z. Wen
%  Version 1.0 .... 2017/8
%
%  Version 1.1 .... 2019/9


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

fun_extra = opts.fun_extra;

%---- If no initial point x is given by the user, generate one at random.
if ~exist('x', 'var') || isempty(x)
    x = M.rand();
end


%--------------------------------------------------------------------------
% Set default parameters for initial solver
if record > 1
    opts_init.record = 1;
else
    opts_init.record = 0;
end
solver_init = @RGBB;
opts_init.tau = 1e-3;
opts_init.maxit = 2000;
opts_init.gtol  = opts.gtol*1e3;
opts_init.xtol  = opts.xtol*1e2;
opts_init.ftol  = opts.ftol*1e2;
% -------------------------------------------------------------------------

% out.nfe = 1;
timetic = tic();
% ------------
% Initialize solution and companion measures: f(x), fgrad(x)

if ~isempty(solver_init)
    if strcmp(func2str(solver_init), 'RGBB') || strcmp(func2str(solver_init), 'RAdaGBB')
        if opts_init.record
            fprintf('initial solver\n');
        end
        t1 = tic; [x,~, outin] = feval(solver_init, x, fun, M, opts_init); t2 = toc(t1);
    else
        t1 = tic; [x,~, outin] = feval(solver_init, x, fun, opts_init); t2 = toc(t1);
    end
    
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
% Set default parameters for subproblems in ARNT
if record > 1
   opts_sub.record = 1;
else
   opts_sub.record = 0; 
end
solver_sub  = @RNewton;
opts_sub.maxitvec  = [100,150,200,300,500];
opts_sub.gtol   = opts.gtol*1e0;
opts_sub.xtol   = opts.xtol*1e0;
opts_sub.ftol   = opts.ftol*1e0;
opts_sub.recordFile = opts.recordFile;
stagnate_check = 0;

% prepare for recording iter. info.
if record || hasRecordFile
    str1 = '    %6s';
    stra = ['\n %8s','%13s',str1,str1,str1,str1,str1,str1];
    str_head = sprintf(stra,...
        'iter', 'F', 'nrmG', 'XDiff', 'FDiff', 'mDiff', 'ratio', 'tau');
    str1 = '  %3.2e';
    str_num = ['(%3d,%3d)','  %14.8e', str1,str1,str1,str1,str1,str1];
    if record
        fprintf('switch to ARNT method \n');
        fprintf('%s\n', str_head);
    end
end

% record iter. info. as a file
if hasRecordFile
    fprintf(fid,'%s\n', str_head);
end

% main loop
for iter = 1:maxit
    
    % set the no. of maximal iter. and stagnate check
    if nrmG >= 1
        opts_sub.maxit = opts.opts_sub.maxit(1);
        stagnate_check = max(stagnate_check,10);
    elseif nrmG >= 1e-2
        opts_sub.maxit = opts_sub.maxitvec(2);
        stagnate_check = max(stagnate_check,20);
    elseif nrmG >= 1e-3
        opts_sub.maxit = opts_sub.maxitvec(3);
        stagnate_check = max(stagnate_check,50);
    elseif nrmG >= 1e-4
        opts_sub.maxit = opts_sub.maxitvec(4);
        stagnate_check = max(stagnate_check,80);
    else
        opts_sub.maxit = opts_sub.maxitvec(5);
        stagnate_check = max(stagnate_check,100);
    end
    opts_sub.stagnate_check = stagnate_check;
    
    % criterion of PCG
    opts_sub.gtol = max(min(0.1*nrmG, 0.1),gtol); % tol for subproblem
    rreg = 10 * max(1, abs(Fp)) * eps; % safeguarding
    if usenumstab
        opts_sub.rreg = rreg;
    else
        opts_sub.rreg = 0;
    end
    
    % subproblem solving
    
    % store the information of current iteration
    TrRho = tau;
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
        [x, ~, out_sub]= feval(solver_sub, xp, Gep, Gp, hess, M, opts_sub);
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
    
    % compute the ratio
    model_decreased = mdiff > 0;
    if ~model_decreased % if the model didn't decrase, ratio = -1
        out.flag(iter) = -6;
        ratio = -1;
    else
        if (abs(redf)<=eps && abs(mdiff)<= eps) || redf == mdiff
            ratio = 1;
        else
            ratio = redf/mdiff;
        end
    end
    
    % compute the difference between iterates and function values
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
        opts_sub.usezero = 0;
    end
    out.nrmGvec(iter) = nrmG;
    out.getavec(iter) = out_sub.geta;
    out.fvec(iter) = F;
    
    % step information in solving subproblem
    stepinf = '';
    switch out.flag(iter)
        case{-1}
            stepinf = '\t[Neg]\n';
        case{-2}
            stepinf = '\t[Stagnate]\n';
        case{-3}
            stepinf = '\t[-Small rho]\n';
        case{-4}
            stepinf = '\t[-Small pHp]\n';
        case{-5}
            stepinf = '\t[Increase in PCG]\n';
        case{-6}
            stepinf = '\t[Increase in model]\n';
        case{1}
            stepinf = '\t \n';
    end
    
    
    % ---- record ----
    if record
        fprintf(strcat(str_num, stepinf),...
            iter, out_sub.iter, Fp, nrmG, XDiff, FDiff, mdiff, ratio, tau);
    end
    
    % record as a file
    if hasRecordFile
        fprintf(fid, strcat(str_num, stepinf), ...
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
    
    % if gradient step is used in solving subproblem in recent 3 steps, we
    % swith to Riemannian gradient type methods
    if iter > 3
        if sum(out.iter_sub(iter-2: iter)) <= 6
            if ~isempty(solver_init)
                opts.maxit = opts.maxit - iter;
                opts.record = 0;
                if strcmp(func2str(solver_init), 'RGBB') || strcmp(func2str(solver_init), 'RAdaGBB')
                    if record
                        fprintf('switch to gradient method\n');
                    end
                    [x,~, out2] = feval(solver_init, x, fun, M, opts);
                else
                    [x,~, out2] = feval(solver_init, x, fun, opts);
                end
                
                % record
                stepinf = '\n';
                if record
                    fprintf(strcat(str_num, stepinf),...
                        iter + 1, out2.iter, out2.fval, out2.nrmG, out2.XDiff, out2.FDiff, out2.fval - out2.fval0, 1, tau);
                end
                if hasRecordFile
                    fprintf(fid, strcat(str_num, stepinf),...
                        iter + 1, out2.iter, out2.fval, out2.nrmG, out2.XDiff, out2.FDiff, out2.fval - out2.fval0, 1, tau);
                end
                
                % store te iter. info.
                nrmG = out2.nrmG;
                XDiff = out2.XDiff;
                FDiff = out2.FDiff;
                out.iter_sub(iter + 1) = out2.iter;
                out.nfe_sub(iter + 1) = out2.nfe;
                break;
            end
            
            
        end
    end
    
    
    
end % end outer loop
timetoc = toc(timetic);

% store the iter. info.
out.XDiff = XDiff;
out.FDiff = FDiff;
out.nrmG = nrmG;
out.fval = F;
out.nfe = out.nfe + sum(out.nfe_sub);
out.avginit = sum(out.iter_sub)/iter;
out.iter = length(out.iter_sub);
out.time = timetoc;

end




