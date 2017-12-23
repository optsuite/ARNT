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
%  Adaptive Regularized Newton Method for Riemannian Optimization
%
% Author: J. Hu, Z. Wen
%  Version 1.0 .... 2017/8

% whether to save the output information, default is 0
dosave = 0;

% choose example
Problist = [1];
for dprob = Problist
    
    if dprob == 1
        nlist = [2000 3000 5000 8000 10000];
        plist = 20;
        alist = 10;
    elseif dprob == 2
        nlist = 5000; alist = 1000;
        plist = [10 20 30 50];
    else
        nlist = 10000; plist = 20;
        alist = [1 10 100 1000];
    end
    
    N_n = length(nlist); N_p = length(plist); N_a = length(alist);
    
    % specify the example and record the results
    filesrc = strcat(pwd,filesep,'results');
    if ~exist(filesrc, 'dir');     mkdir(filesrc);   end
    filepath = strcat(filesrc, filesep, 'noneig');
    if ~exist(filepath, 'dir');    mkdir(filepath);  end

    if dosave
        filename = strcat(filepath,filesep,'Date_',...
            num2str(date),'noneig_',num2str(dprob),'.txt');
        fid = fopen(filename,'w+');
        fprintf(fid,'\n\n');
        fprintf(fid,'& \t \\multicolumn{3}{c|}{GBB} & \t \\multicolumn{3}{c|}{ARNT} & \t \\multicolumn{3}{c|}{RTR}');
        fprintf(fid,'\\\\ \\hline \n');
        fprintf(fid,'Prob \t & \t its & \t nrmG &\t time & \t its & \t nrmG &\t time & \t its & \t nrmG &\t time');
        fprintf(fid,'\\\\ \\hline \n');
    end
    
    % loop
    for ni = 1:N_n
        for np = 1:N_p
            for na = 1:N_a
                
                seed = 2010;
                if exist('RandStream','file')
                    RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
                else
                    randrot('state',seed); randn('state',seed);
                end
                
                p = plist(np);
                n = nlist(ni);
                alpha = alist(na);
                
                fprintf('------- (n,p,alpha) = (%d, %d, %.1f)----\r\n',n,p, alpha);
                name = strcat('noneig-','n-',num2str(n),'-p-',num2str(p),'-al-',num2str(alpha));
                
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
                
                % Options for solvers
                opts.record = 1;
                opts.usenumstab = 0;
                opts.tau = 10; % regularized parameter tau
                opts.maxit = 500;                
                opts.gtol  = 1e-6;
                opts.xtol  = 0e-10;
                opts.ftol  = 0e-16;                
                M = stiefelfactory(n,p);
                opts.hess = @hess;
                opts.grad = @grad;
                opts.fun_TR = @fun_sub;
                opts.fun_extra = @fun_extra;
                
                % Options for initial solvers
                opts.opts_init.record = 0;
                opts.solver_init = @RGBB;
                opts.opts_init.tau   = 1e-3;
                opts.opts_init.maxit = 2000;
                opts.opts_init.gtol  = opts.gtol*1e3;
                opts.opts_init.xtol  = opts.xtol*1e2;
                opts.opts_init.ftol  = opts.ftol*1e2;
           
                % Options for subsolvers
                opts.opts_sub.record = 0;
                opts.solver_sub  = @RNewton;
                opts.opts_sub.tau    = 1e-3;
                opts.opts_sub.maxit  = [100,150,200,300,500];
                opts.opts_sub.gtol   = opts.gtol*1e0;
                opts.opts_sub.xtol   = opts.xtol*1e0;
                opts.opts_sub.ftol   = opts.ftol*1e0;
                
                recordname = strcat(filepath,filesep,'Date_',...
                    num2str(date),'noneig','n',num2str(n),'p',num2str(p),'alpha',num2str(alpha),'.txt');
                opts.recordFile = recordname;
                opts.opts_init.recordFile = opts.recordFile;
                opts.opts_sub.recordFile = opts.recordFile;
                
                t0 = tic; [~,~, out_ARNT] = arnt(X0, @fun, M, opts); tsolve_ARNT = toc(t0);
                
                if dosave
                    save(strcat(filepath,filesep,'ARNT-',name), 'out_ARNT', 'tsolve_ARNT');
                end
                
                OutIter_ARNT = out_ARNT.iter;
                InnerIter_ARNT = sum(out_ARNT.iter_sub);
                nfe_ARNT = out_ARNT.nfe;
                f_ARNT = out_ARNT.fval;
                nrmG_ARNT = out_ARNT.nrmG;
                
                
                fprintf('ARNT|  f: %8.6e, nrmG: %2.1e, cpu: %4.2f, OutIter: %3d, InnerIter: %4d, nfe: %4d,\n',...
                    f_ARNT, nrmG_ARNT, tsolve_ARNT, OutIter_ARNT, InnerIter_ARNT, nfe_ARNT);
                
                if dosave
                    fprintf(fid,'ARNT & %14.8e &%4d(%4.0f) & \t %1.1e &\t %8.1f \\\\ \\hline \n', ...
                        f_ARNT, OutIter_ARNT, InnerIter_ARNT/OutIter_ARNT, nrmG_ARNT, tsolve_ARNT);
                end
                
            end
        end
        
    end
    if dosave; fclose(fid); end
end


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

