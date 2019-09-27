function test_noneig
% simple nonlinear eigenvalue problems: a simplified model problem for
% density functional theory
% min 0.5*Tr(X'*L*X) + alpha/4*rho(X)'*L^{dag}*rho(X), s.t. X'*X = I_k
% X is n by p matrix, L is a symmetric matrix and rho(X) s a vector whose
% components are the diagonal elements of X*X'
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

% test problem setting
% nlist = [2000 3000 5000 8000 10000];
% plist = [10 20 30 50];
% alist = [1 10 100 1000];
nlist = 2000; plist = 10; alist = 100;

% whether to save the output information, default is 0
dosave = 1;

% set tolerance for ARNT
gtol = 1e-12;

% specify the example and record the results
filesrc = strcat(pwd,filesep,'results');
if ~exist(filesrc, 'dir');     mkdir(filesrc);   end
filepath = strcat(filesrc, filesep, 'noneig');
if ~exist(filepath, 'dir');    mkdir(filepath);  end

if dosave
    filename = strcat(filepath,filesep,'Date_',...
        num2str(date),'noneig','.txt');
    fid = fopen(filename,'w+');
    fprintf(fid,'\n');
    fprintf(fid,' & \t \\multicolumn{4}{c|}{ARNT}');
    fprintf(fid,'\\\\ \\hline \n');
    fprintf(fid,'Prob \t & fval \t  & \t  its \t & \t nrmG    &\t time');
    fprintf(fid,'\\\\ \\hline \n');
end

% loop
for n = nlist
    for p = plist
        for alpha = alist
            
            seed = 2010;
            if exist('RandStream','file')
                RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
            else
                randrot('state',seed); randn('state',seed);
            end
            
            fprintf('\n------- (n,p,alpha) = (%d, %d, %.1f)----\n',n,p, alpha);
            
            % generate L
            L = gallery('tridiag', n, -1, 2, -1);
            [Ll,Lu] = lu(L);
            
            % intial point
            X = randn(n, p);
            [U, ~, V] = svd(X, 0);
            X_init = U*V';
            tempM2 = alpha*(L\(sum(X_init.^2,2)));
            tempM2 = spdiags(tempM2,0,n,n);
            tempM = L + tempM2;
            [U0, ~, ~] = eigs(tempM, p,'sm');
            X0 = U0;
            
            % Stiefel manifold
            M = stiefelfactory(n,p);
            
            % identify Eulcidean gradient and Hessian
            opts.hess = @hess;
            opts.grad = @grad;
            opts.fun_extra = @fun_extra;
            
            % set default parameters for ARNT
            opts.record = 1; % 0 for slient, 1 for outer iter. info., 2 or more for all iter. info.
            opts.xtol = 1e-12;
            opts.ftol = 1e-12;
            opts.gtol = gtol;
            opts.maxit = 500;
            opts.tau = 1;
            opts.usenumstab = 1;
            
            % run ARNT
            recordname = strcat(filepath,filesep,'Date_',...
                num2str(date),'noneig','n',num2str(n),'p',num2str(p),'alpha',num2str(alpha),'.txt');
            opts.recordFile = recordname;
            t0 = tic; [~,~, out_ARNT] = arnt(X0, @fun, M, opts); tsolve_ARNT = toc(t0);
                        
            % print info. in command line
            fprintf('ARNT|  f: %8.6e, nrmG: %2.1e, cpu: %4.2f, OutIter: %3d, InnerIter: %4d, nfe: %4d,\n',...
                out_ARNT.fval, out_ARNT.nrmG, tsolve_ARNT, out_ARNT.iter, sum(out_ARNT.iter_sub), out_ARNT.nfe);
            
            % save info.
            if dosave
                name = strcat('noneig-','n-',num2str(n),'-p-',num2str(p),'-al-',num2str(alpha));
                save(strcat(filepath, filesep,'ARNT-',name), 'out_ARNT', 'tsolve_ARNT');
                fprintf(fid,'ARNT & %14.8e &%4d(%4.0f)    & \t %1.1e &\t %6.1f \\\\ \\hline \n', ...
                    out_ARNT.fval, out_ARNT.iter, sum(out_ARNT.iter_sub)/out_ARNT.iter, out_ARNT.nrmG, tsolve_ARNT);
            end
            
        end
    end
    
end

if dosave; fclose(fid);  end


    function [f,g] = fun(X,~)
        LX = L*X;
        rhoX = sum(X.^2, 2); % diag(X*X');
        tempa = Lu\(Ll\rhoX); tempa = alpha*tempa;
        
        f = 0.5*sum(sum(X.*(LX))) + 1/4*(rhoX'*tempa);
        g = LX + bsxfun(@times,tempa,X);
    end

    function data = fun_extra(data)
        
        XP = data.XP;
        data.rhoX = sum(XP.^2,2);
        
    end

    function g = grad(X)
        rhoX = sum(X.^2, 2); % diag(X*X');
        tempa = Lu\(Ll\rhoX); tempa = alpha*tempa;
        g = L*X + bsxfun(@times,tempa,X);
    end


    function h = hess(X, U, data)
        
        rhoX = data.rhoX;
        rhoXdot = 2*sum(X.*U, 2);
        tempa = Lu\(Ll\rhoXdot);
        tempa = alpha*tempa;
        tempb = Lu\(Ll\rhoX);
        tempb = alpha*tempb;
        if isfield(data,'sigma')
            h = L*U + bsxfun( @times,tempa,X) + bsxfun(@times, tempb + data.sigma, U);
        else
            h = L*U + bsxfun( @times,tempa,X) + bsxfun(@times, tempb, U);
        end
    end
end

