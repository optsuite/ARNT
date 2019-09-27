function [x, out] = RNewton(x0, egrad, rgrad, H, M, opts)
% modified Newton method to solve the subproblem
% min <egrad, x - x_k> + 1/2*<H[x - x_k], x - x_k> 
%   + 1/2*sigma_k \|x -  x_k\|^2
% 
% Input:
%         x0 --- initial guess
%      egrad --- Euclidean gradient of original function at x_k
%      rgrad --- Riemannian gradient of original function at x_k
%          H --- Euclidean Hessian of original function at x_k
%          M --- manifold
%       opts --- option structure with fields
%                gtol         the accuracy for solving the Newton direction
%                eta, rhols   parameters in line search
%                maxit        max number of iterations  
%                record       = 0, no print out
%
% Output: 
%          x --- soultion
%        out --- output information
%
% -----------------------------------------------------------------------
% Reference: 
%  J. Hu, A. Milzark, Z. Wen and Y. Yuan
%  Adaptive Quadratically Regularized Newton Method for Riemannian Optimization
%
% Author: J. Hu, Z. Wen
%  Version 1.0 .... 2017/8
%
%  Version 1.1 .... 2019/9


% termination rule
if ~isfield(opts, 'gtol');       opts.gtol = 1e-7;   end % 1e-5
if ~isfield(opts, 'eta');        opts.eta  = 0.2;     end % 1e-5
if ~isfield(opts, 'rhols');      opts.rhols  = 1e-4;   end
if ~isfield(opts, 'maxit');      opts.maxit  = 200;    end
if ~isfield(opts, 'usezero');    opts.usezero = 1;      end
if ~isfield(opts, 'record');     opts.record = 0;      end

% copy parameters
rhols = opts.rhols;  eta = opts.eta; gtol = opts.gtol; record = opts.record;

inner = M.inner;

% numerical stability
if isfield(opts, 'rreg')
    rreg = opts.rreg;
else
    rreg = 0;
end

% record iter. info.
hasRecordFile = 0;
if 0; isfield(opts, 'recordFile')
    fid = fopen(opts.recordFile,'a+'); hasRecordFile = 1;
end

% data structure of x0
if isstruct(x0)
    matX0 = x0.matX;
end

if record || hasRecordFile
    str1 = '  %6s';
    stra = ['\n %10s',str1,' %2s',str1,str1,str1,str1,str1,str1,str1,'\n'];
    str_head = sprintf(stra,...
        'f', 'iters', 'flag', 'tol', '    geta',' angle',' err','pHp','nls','step');
    str1 = '  %1.1e';
    str_num = ['%1.6e','    %d','   %d  ', str1,str1,'  %3.0f  ',str1,str1,'   %d ',str1,'\n'];
end


if hasRecordFile
    fprintf(fid,stra, ...
        'f', 'iters', 'flag', 'tol', 'geta','angle','err','pHp','nls','step');
end

% set the initial iter. no.
out.nfe = 0; out.fval0 = 0;
% pcg; return -1 if the Hessian is indefinite
[deta,negcur,iters,err,pHp,resnrm,flag,mark,out_cg] ...
    = pcgw(x0,H,M,egrad,M.lincomb(x0,-1,rgrad),opts,gtol);

% record the iter. info.
out.nfe = out.nfe + iters;
out.iter = iters;
out.flag = flag;
out.tau = mark;
out.deta = deta;

%%%%%%%%%%%%%%%%%%%%%%%%

% judge whether to use the negative direction
% If a not too small negative curvature is encoutered, 
% combine it to get a new direction
if mark > 1e-10
    gcur = inner(x0, rgrad,negcur);
    % negcur = -sign(gcur) * negcur;
    deta = M.lincomb(x0, 1, deta, gcur/pHp, negcur);
end

% inner product between gradient and descent direction
geta = inner(x0, rgrad,deta);
angle = acos(geta/M.norm(x0,rgrad)/M.norm(x0,deta))/pi*180;

out.geta = geta;
out.angle = angle;

% Armijo search with intial step size 1
nls = 1; deriv = rhols*geta; step = 1;
while 1
    x = M.retr(x0, deta, step);
    if isstruct(x)
        matX = x.U*x.S*x.V';
        x.matX = matX;
        xx0 = matX - matX0;
        Hxx0 = H(xx0);
    else
        xx0 = x - x0;
        Hxx0 = H(xx0);
    end
    
    f = iprod(egrad,xx0) + .5*iprod(Hxx0,xx0);
    out.nfe = out.nfe + 1;

    if f - rreg <= step*deriv || nls >= 10
        break;
    end
    step = eta*step;
    nls = nls + 1;
end

% print iter. info.
out.fval = f; opts.record = 1;

if record
    fprintf('%s', str_head);    
    fprintf(str_num, ...
        full(f), iters, flag, gtol, geta, angle, err, pHp, nls, step);  
end

if hasRecordFile
    fprintf(fid, str_num, ...
        full(f), iters, flag, gtol, geta, angle, err, pHp, nls, step);
end

end

function a = iprod(x,y)
a = real(sum(sum(conj(x).*y)));
%a = sum(sum(x.*y));
end


function  q = precondfun(r)

precond = 0;
%if isfield(par,'precond'); precond = par.precond; end

if (precond == 0)
    q = r;
    %    elseif (precond == 1)
    %       q = L.invdiagM.*r;
    %    elseif (precond == 2)
    %      if strcmp(L.matfct_options,'chol')
    %         q(L.perm,1) = mextriang(L.R, mextriang(L.R,r(L.perm),2) ,1);
    %      elseif strcmp(L.matfct_options,'spcholmatlab')
    %         q(L.perm,1) = mexbwsolve(L.Rt,mexfwsolve(L.R,r(L.perm,1)));
    %      end
    %      if isfield(par,'sig')
    %         q = q/par.sig;
    %      end
end
end

function [deta,negcur, iter,err,pHp,resnrm, flag,mark, out] ...
    = pcgw(x,H,M,grad,r,opts,tol)
if ~isfield(opts, 'maxit');  opts.maxit  = 200;   end
if ~isfield(opts, 'minit');  opts.minit  = 1;   end
if ~isfield(opts, 'stagnate_check');  opts.stagnate_check  = 50;   end
if ~isfield(opts, 'record'); opts.record  = 0;   end
if ~isfield(opts, 'usezero'); opts.usezero  = 1;   end

% copy parameters
maxit = opts.maxit;
minit = opts.minit;
stagnate_check = opts.stagnate_check;
record = 0;
zero = M.zerovec(x); % zero element in the tangent space
mark = 0; f_CG = 0;
alpha = 0; beta = 0;

% initial point
usezero = 1;
if usezero
    deta = zero;
    Hdeta = zero;
else
    deta = opts.deta;
    Hdeta = H(deta);
    Hdeta = M.ehess2rhess(x,grad,Hdeta,deta);
    r = r - Hdeta;
    deta = zero;
end

% set the initial iter. no.
r0 = r; z = precondfun(r);
p = z;
err = M.norm(x, r); resnrm(1) = err; minres = err;
rho = M.inner(x,r,z);
negcur = zero;
flag = 1;

if record
    str1 = '   %6s';
    stra = ['%6s', str1, str1, str1,'\n'];
    str_head = sprintf(stra,...
        'iter', 'alpha', 'pHp', 'err');
    str1 = '   %1.2e';
    str_num = ['%d', str1,str1,str1,'\n'];
end

% PCG loop
for iter = 1:maxit
    % Euclidean Hessian
    Hp = H(p);
    % Riemannian Hessian
    rHp = M.ehess2rhess(x,grad,Hp,p);
    pHp = M.inner(x, p, rHp);
    
    nrmp = M.norm(x,p)^2;
    scalenrmp = 1e-10*nrmp;
    
    % check the stopping criterion, construct the new direction if stopped
    if pHp <= scalenrmp      
        if iter == 1; deta = p;
        else
            if  pHp <= -scalenrmp 
                negcur = p; mark = -pHp/nrmp; 
                
            end
        end
        flag = -1; out.msg = 'negative curvature';
        break;
    end
        
    if abs(pHp) < 1e-30
             
        flag = -4;
        break;
    else
        alpha = rho/pHp;
        
        if iter == 1
            deta_new = M.lincomb(x,alpha,p);
            Hdeta_new = M.lincomb(x,alpha,rHp);
        else
            deta_new = M.lincomb(x,1,deta, alpha ,p);
            Hdeta_new = M.lincomb(x,1,Hdeta,alpha,rHp);
        end
        f_CG_new = - M.inner(x,deta_new,r0) + .5*M.inner(x,deta_new,Hdeta_new);
        
        if f_CG_new > f_CG
            out.msg = 'no decrease in PCG';
            flag = -5;
            break; 
        else
            f_CG = f_CG_new;
            deta = deta_new; Hdeta = Hdeta_new;
        end
        r = M.lincomb(x, 1, r ,-alpha, rHp);
    end
    
    % residual
    err = M.norm(x,r); resnrm(iter+1) = err;
    
    if record 
        if iter == 1
            fprintf('\n%s', str_head);
        end
        fprintf(str_num, ...
            iter, alpha, pHp, err);
    end
    
    % check stagnate and stopping criterion
    if (err < minres); minres = err; end
    if (err < tol) && (iter > minit)
        out.msg = 'accuracy'; 
        flag = 1;
        break; 
    end
    if (iter > stagnate_check) && (iter > 10)
        ratio = resnrm(iter-9:iter+1)./resnrm(iter-10:iter);
        if (min(ratio) > 0.97) && (max(ratio) < 1.03)
            flag = -2;
            out.msg = 'stagnate check';
            break;
        end
    end
    %%-----------------------------
    if abs(rho) < 1e-30
        flag = -3;
        out.msg = 'rho is small';
        break;
    else
        z = precondfun(r);
        rho_old = rho;
        rho = M.inner(x,r,z);
        beta = rho/rho_old;
        p = M.lincomb(x,1,z, beta, p);
        
        % If the tangent vector p is not represented by coordinates, we use
        % re-projection to improve consistency
        if ~isstruct(x)
           p = M.proj(x,p);
        end
    end

end
out.alpha = alpha;
out.beta = beta;
end

