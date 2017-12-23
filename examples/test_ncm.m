function test_ncm
% low rank nearest correlation matrix estimation
% min .5*||H.*(X - G)||_F^2, s.t. X_ii = 1, rank(X) <= p. X psd
% H is a nonnegative symmetric weight matrix, G is a symmetric matrix, X, G
% and H are all n by n matrices.

% equivalent form (the formulation we solved below)
% X = V'V, V = [V_1, ..., V_n], V is a p by n matrix
% min .5*||H.*(V'*V - G)||_F^2, s.t. ||V_i|| = 1, i = 1, ..., n  
%
% Data 'Eg5.5H' and 'Eg5.5C' can be download from 
% http://www.grouplens.org/node/73, the other data set 'Leukemia' is
% referred from Li, Lu, and Kim-Chuan Toh. "An inexact interior point method 
% for l_1-regularized sparse covariance selection." Mathematical 
% Programming Computation 2.3 (2010): 291-315. 
% 
% -----------------------------------------------------------------------
% Reference: 
%  J. Hu, A. Milzark, Z. Wen and Y. Yuan
%  Adaptive Regularized Newton Method for Riemannian Optimization
%
% Author: J. Hu, Z. Wen
%  Version 1.0 .... 2017/8



% choose examples
opts = []; seed = 2010;
% Problist = [1:11];
Problist = [1]; 

% choose rank
plist = [5, 10, 20,  50, 100, 150, 200];
plist = [20];

% whether to save the output information, default is 0
dosave = 0;

for dprob = Problist
    
    % fix seed
    if exist('RandStream','file')
        RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
    else
        randrot('state',seed); randn('state',seed);
    end
    
    % save iter. info.
    filesrc = strcat(pwd,filesep,'results');
    if ~exist(filesrc, 'dir');     mkdir(filesrc);   end
    filepath = strcat(filesrc, filesep, 'ncm');
    if ~exist(filepath, 'dir');    mkdir(filepath);  end
    
    if dosave
        filename = strcat(filepath,filesep,'Date_',...
            num2str(date),'ncm_',num2str(dprob),'.txt');
        
        fid = fopen(filename,'w+');
        
        fprintf(fid,'\n');
        % fprintf(fid,'& \t \\multicolumn{3}{c}{OptM_GBB} & \t \\multicolumn{3}{c}{ARNT} & \t \\multicolumn{3}{c}{RTR}  & \t \\multicolumn{3}{c}{GBB}');
        fprintf(fid,'& \t \\multicolumn{3}{c|}{GBB} \t \\multicolumn{3}{c|}{AdaGBB} & \t \\multicolumn{3}{c|}{ARNT}  & \t \\multicolumn{3}{c|}{RTR}');
        
        fprintf(fid,'\\\\ \\hline \n');
        % fprintf(fid,'Prob \t & \t iter & \t nrmG &\t time & \t iter(in) & \t nrmG &\t time & \t iter(in) & \t nrmG &\t time & \t iter(in) & \t nrmG &\t time');
        fprintf(fid,'Prob \t & \t its & \t nrmG &\t time & \t its & \t nrmG &\t time & \t its & \t nrmG &\t time & \t its & \t nrmG &\t time');
        
        fprintf(fid,'\\\\ \\hline \n');
    end
    
    % matrix G
    switch dprob
        case {1, 2}
            n = 500; a = 1:n;
            II = repmat(a', 1, n);
            JJ = repmat(a,  n, 1);
            G = 0.5 + 0.5*exp( -0.05*abs(II - JJ) );
        case {3, 4}
            load('Leukemia.mat', 'S');
            G = S;
            n = size(G,1);
        case {5, 6}
            load('ER.mat', 'S');
            G = S;
            n = size(G,1);
        case {7, 8}
            load('hereditarybc.mat', 'S');
            G = S;
            n = size(G,1);
        case {9, 10}
            load('Lymph.mat', 'S');
            G = S;
            n = size(G,1);
            
        case {11}
            load('Eg5.5C.mat', 'G');
            n = size(G,1);
    end
    
    % Generate weight matrix H
    H = [];
    switch dprob
        case {2, 4, 6, 8, 10}
            if exist('RandStream','file')
                RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
            else
                randrot('state',seed); randn('state',seed);
            end
            
            H0 = sprand(n,n,0.5);
            H0 = triu(H0) + triu(H0,1)'; % W0 is likely to have small numbers
            H0 = (H0 + H0')/2;
            H0 = 0.01*ones(n,n) + 99.99*H0; %%% H0 is in [0.01, 100]
            H1 = rand(n,n);
            H1 = triu(H1) + triu(H1,1)'; % W1 is likely to have small numbers
            H1 = (H1 + H1')/2;
            H  = 0.1*ones(n,n) + 9.9*H1; %%% H is in [.1,10]
            %%%%%%%%%%%%%%%%%%%%% Assign weights H0  on partial elemens
            s = sprand(n,1,min(10/n,1));
            I = find(s>0);
            d = sprand(n,1,min(10/n,1));
            J = find(d>0);
            if ~isempty(I) >0 && ~isempty(J)>0
                H(I,J) = H0(I,J);
                H(J,I) = H0(J,I);
            end
            H = (H + H')/2;
            opts.randH = 1;
            %%%%%%%%%%%% end of  assignings weights from one only on partial elemens
        case {11}
            load('Data/Eg5.5H.mat', 'H');
            opts.randH = 0;
    end
    H2 = H.*H;
    
    
    % test different rank p
    for p = plist
        
        % fixed seed
        if exist('RandStream','file')
            RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
        else
            randrot('state',seed); randn('state',seed);
        end
        
        % initial point
        x0 = randn(p,n);        
        nrmx0 = dot(x0,x0,1);
        x0 = bsxfun(@rdivide, x0, sqrt(nrmx0));
        
        
        opts.record = 1;
        opts.gtol = 1e-6;
        opts.xtol = 0e-5;
        opts.ftol = 0e-8;        
        M = obliquefactory(p,n);
        opts.hess = @hess;
        opts.grad = @grad;
        opts.fun_TR = @fun_sub;
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
        opts.fun_TR = @fun_sub;
        opts.gtol = 1e-6;
        opts.maxit = 500;
        opts.tau = 10;
        opts.usenumstab = 1;
        recordname = strcat(filepath,filesep,'ARNT_','Date_',...
            num2str(date),'ncm','n',num2str(n),'p',num2str(p),'.txt');
        opts.recordFile = recordname;
        opts.opts_init.recordFile = opts.recordFile;
        opts.opts_sub.recordFile = opts.recordFile;
        t = tic; [~, ~, out_ARNT] = arnt(x0, @fun, M, opts); tsolve_ARNT = toc(t);
        
        if dosave
            name = strcat('ncm-','n-',num2str(n),'-p-',num2str(p));
            save(strcat(filepath, filesep,'ARNT-',num2str(dprob),'-',name), 'out_ARNT', 'tsolve_ARNT');
        end
        
        OutIter_ARNT = out_ARNT.iter;
        InnerIter_ARNT = sum(out_ARNT.iter_sub);
        nfe_ARNT = out_ARNT.nfe;
        f_ARNT = out_ARNT.fval;
        nrmG_ARNT = out_ARNT.nrmG;
        
        fprintf(1, 'ARNT|  f: %8.6e, nrmG: %2.1e, cpu: %4.2f, OutIter: %3d, InnerIter: %4d, nfe: %4d,\n',...
            f_ARNT, nrmG_ARNT, tsolve_ARNT, OutIter_ARNT, InnerIter_ARNT, nfe_ARNT);
        
        if dosave
            name = num2str(p);
            fprintf(fid,'%10s & \t %d(%4.1f) & \t %1.1e &\t %.2f',...
                name, OutIter_ARNT,InnerIter_ARNT/OutIter_ARNT, nrmG_ARNT, tsolve_ARNT);
            fprintf(fid,'\\\\ \\hline \n');
        end
        
    end
    
    if dosave; fclose(fid);  end
    
end

    function [f,g] = fun(X,~)
        XtX = X'*X;
        if ~exist('H','var') || isempty(H)
            gg = XtX - G;
            f =sum(sum(gg.^2))/2;
            g = 2*(X*gg);
        else
            gg = H.*(XtX - G);
            f = sum(sum(gg.^2))/2;
            g = 2*(X*(H.*gg));
        end
    end

    function [f,g] = fun_sub(X,data)
        
        XP = data.XP;
        tau = data.TrRho;
        U = X - XP;
        %         norm(data.XtX,'fro')
        UtX = U'*XP;
        if isfield(data,'XtX')
            gg = data.gg;
            tmpG = data.G;
        else
            XtX = XP'*XP;
            if ~exist('H','var') || isempty(H)
                gg = XtX - G;
                tmpG = 2*XP*gg;
            else
                gg = H.*(XtX - G);
                tmpG = 2*(XP*(H.*gg));
            end
            data.XtX = XtX;
            data.gg = gg;
            data.G = tmpG;
        end
        
        if ~exist('H','var') || isempty(H)
            tmp1 = U*gg;
            tmp2 = XP*(UtX + UtX');
        else
            tmp1 = U*(H.*gg);
            tmp2 = XP*(H.*(H.*(UtX + UtX')));
        end
        h = 2*(tmp1 + tmp2);
        nrm = norm(U,'fro');
        f = sum(sum(U.*tmpG)) + .5*sum(sum(h.*U)) + 1/3*nrm^3*tau;
        g = tmpG + h + tau*nrm*U;
        %         if narargout > 1
        %             g = tmpG + h + tau*nrm*U;
        %         end
    end

    function data = fun_extra(data)
        
        XP = data.XP;
        data.XtX = XP'*XP;
        if ~exist('H','var') || isempty(H)
            
            data.gg = data.XtX - G;
            data.G = 2*XP*data.gg;
        else
            data.gg = H.*(data.XtX - G);
            data.G = 2*(XP*(H.*data.gg));
        end
    end

    function g = grad(X)
        XtX = X'*X;
        if ~exist('H','var') || isempty(H)
            gg = (XtX - G);
            g = 2*(X*gg);
        else
            gg = H.*(XtX - G);
            g = 2*(X*(H.*gg));
        end
    end


    function h = hess(X, U, data)
        
        gg = data.gg;
        UtX = U'*X;
        if ~exist('H','var') || isempty(H)
            
            tmp1 = U*(gg);
            tmp2 = X*(UtX + UtX');
        else
            
            tmp1 = U*(H.*gg);
            tmp2 = X*(H2.*(UtX + UtX'));
        end
        h = 2*(tmp1 + tmp2) + data.sigma*U;
        
    end

end
