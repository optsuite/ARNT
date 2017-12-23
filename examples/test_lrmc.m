function test_lrmc
% low rank matrix completion
% Given a partially observed m by n matrix A, denote the observed entries
% index by matrix P, find the lowest-rank matrix to fit A. The problem can
% be split into a series fixed rank problem as
% min 0.5*||P.*X - A||_F^2, s.t. rank(X) = k.
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

% test problem setting
% nlist = [1000, 2000, 4000, 8000];
% klist = [10, 20, 30, 40, 50, 60];
% fraclist = [0.1, 0.2, 0.5, 0.8, 2, 3];
nlist = 1000; klist = 10; fraclist = 0.1;

% store the numerical results
filesrc = strcat(pwd,filesep,'results');
if ~exist(filesrc, 'dir');     mkdir(filesrc);   end
filepath = strcat(filesrc, filesep, 'lrmc');
if ~exist(filepath, 'dir');     mkdir(filepath);   end

if dosave
    filename = strcat(filepath,filesep,'Date_',...
        num2str(date),'lrmc','.txt');
    
    fid = fopen(filename,'w+');
    
    fprintf(fid,'\n\n');
    fprintf(fid,'& \t \\multicolumn{3}{c|}{GBB} & \t \\multicolumn{3}{c|}{ARNT} & \t \\multicolumn{3}{c|}{RTR}');
    fprintf(fid,'\\\\ \\hline \n');
    fprintf(fid,'Prob \t & \t its & \t nrmG &\t time & \t its & \t nrmG &\t time & \t its & \t nrmG &\t time');
    fprintf(fid,'\\\\ \\hline \n');
end

% main loop
for n = nlist
    for k = klist
        for frac = fraclist
            
            seed = 2011;
            fprintf('seed: %d\n', seed);
            if exist('RandStream','file')
                RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
            else
                randrot('state',seed); randn('state',seed^2);
            end
            
            % Generate m x n low-rank matrix
            m = n;
            L = randn(m, k);
            R = randn(n, k);
            A = L*R';
            % Generate index for observed entries: P(i, j) = 1 if the entry
            % (i, j) of A is observed, and 0 otherwise.
            
            fraction = frac * k*(m+n-k)/(m*n);
            P = sparse(rand(m, n) <= fraction);
            % partially observed matrix is PA:
            PA = P.*A;
            
            % manifold of matrices of size mxn of fixed rank k.
            M = fixedrankfactory(m, n, k);
            
            % Compute an initial guess. Points on the manifold are represented as
            % structures with three fields: U, S and V. U and V need to be
            % orthonormal, S needs to be diagonal.
            [U, S, V] = svds(randn(m,n), k);
            X0.U = U;
            X0.S = S;
            X0.V = V;
                        
            % specify name to recode the results
            name = strcat('LRMC-','m-',num2str(m),'-n-',num2str(n),'-k-',num2str(k),'-frac-',strrep(num2str(frac,'%.1f'),'.','-'));
            opts.hess = @hess;
            opts.grad = @grad;
            opts.record = 1;
            opts.xtol = 0;1e-6;
            opts.ftol = 0;
            opts.gtol = 1e-6;
            opts.maxit = 500;
            opts.fun_extra = @fun_extra;
            
            opts.opts_init.record = 1;
            opts.solver_init = @RGBB;
            opts.opts_init.tau   = 1e-3;
            opts.opts_init.maxit = 2000;
            opts.opts_init.gtol  = opts.gtol*1e3;
            opts.opts_init.xtol  = opts.xtol*1e2;
            opts.opts_init.ftol  = opts.ftol*1e2;
            opts.opts_sub.record = 0;
            opts.solver_sub  = @RNewton;
            opts.opts_sub.tau    = 1e-3;
            opts.opts_sub.maxit  = [100,150,200,300,500];
            
            opts.opts_sub.gtol   = opts.gtol*1e0;
            opts.opts_sub.xtol   = opts.xtol*1e0;
            opts.opts_sub.ftol   = opts.ftol*1e0;
            opts.fun_TR = [];
            opts.tau = 1;
            opts.theta = 1;
            
            recordname = strcat(filepath,filesep,'Date', ...
                num2str(date),'lrmc','-m-',num2str(m),'-n-',num2str(n),'-k-',num2str(k),'.txt');
            opts.recordFile = recordname;
            opts.opts_init.recordFile = opts.recordFile;
            opts.opts_sub.recordFile = opts.recordFile;
            
            t0 = tic; [~,~,out_ARNT] = arnt(X0, @fun, M, opts); tsolve_ARNT = toc(t0);
            
            if dosave
                save(strcat(filepath, filesep,'ARNT-',strrep(num2str(frac,'%.1f'),'.','-'),name), 'out_ARNT', 'tsolve_ARNT');
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

if dosave; fclose(fid);  end

    function [f,g] = fun(X,~)
        matX = X.U*X.S*X.V';
        f = .5*norm( P.*matX - PA, 'fro')^2;
        
        g = P.*matX - PA;
    end

    function g = grad(X)
        matX = X.U*X.S*X.V';
        g = P.*matX - PA;
    end

    function data = fun_extra(data)
        
        if isfield(data,'sigma')
            data.sigP = (1 + data.sigma)*P;
        end
    end



    function h = hess(X, eta, data)
      
        if isfield(data, 'sigP')
            sigP = data.sigP;
        end
        
        if isstruct(eta)
            Xdot = M.tangent2ambient(X, eta);
            if isfield(data,'sigP')
                h = sigP.*Xdot;
            else
                h = P.*Xdot;
            end
        else
            if isfield(data,'sigP')
                
                h = sigP.*eta;
            else
                h = P.*eta;
                
            end
        end
    end

end
