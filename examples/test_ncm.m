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
%  Adaptive Quadratically Regularized Newton Method for Riemannian Optimization
%
% Author: J. Hu, Z. Wen
%  Version 1.0 .... 2017/8
%
%  Version 1.1 .... 2019/9

% choose examples
% Problist = [1:11];
Problist = 1;

% choose rank
% plist = [5, 10, 20,  50, 100, 150, 200];
plist = 20;

% whether to save the output information, default is 0
dosave = 1;

% set tolerance for ARNT
gtol = 1e-6;

for dprob = Problist
    
    % fix seed
    seed = 2010;
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
        fprintf(fid,' & \t \\multicolumn{4}{c|}{ARNT}');
        
        fprintf(fid,'\\\\ \\hline \n');
        fprintf(fid,'Prob \t & fval \t  & \t its \t & \t nrmG    &\t time');
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
        
        % obique manifold
        M = obliquefactory(p,n);
        
        % Eulcidean gradient and Hessian
        opts.hess = @hess;
        opts.grad = @grad;
        opts.fun_extra = @fun_extra;
        
        % initial point
        x0 = randn(p,n);
        nrmx0 = dot(x0,x0,1);
        x0 = bsxfun(@rdivide, x0, sqrt(nrmx0));
        
        % set default parameters for ARNT
        opts.record = 1; % 0 for slient, 1 for outer iter. info., 2 or more for all iter. info.
        opts.gtol = gtol;
        opts.xtol = 1e-12;
        opts.ftol = 1e-12;
        opts.maxit = 500;
        opts.tau = 10;
        opts.usenumstab = 1;
        
        % run ARNT
        recordname = strcat(filepath,filesep,'ARNT_','Date_',...
            num2str(date),'ncm','n',num2str(n),'p',num2str(p),'.txt');
        opts.recordFile = recordname;
        t = tic; [~, ~, out_ARNT] = arnt(x0, @fun, M, opts); tsolve_ARNT = toc(t);
        
        % print info. in command line
        fprintf('ARNT|  f: %8.6e, nrmG: %2.1e, cpu: %4.2f, OutIter: %3d, InnerIter: %4d, nfe: %4d,\n',...
            out_ARNT.fval, out_ARNT.nrmG, tsolve_ARNT, out_ARNT.iter, sum(out_ARNT.iter_sub), out_ARNT.nfe);
        
        % save info.
        if dosave
            name = strcat('ncm-','n-',num2str(n),'-p-',num2str(p));
            save(strcat(filepath, filesep,'ARNT-',name), 'out_ARNT', 'tsolve_ARNT');
            fprintf(fid,'ARNT & %14.8e &%4d(%4.0f)    & \t %1.1e &\t %6.1f \\\\ \\hline \n', ...
                out_ARNT.fval, out_ARNT.iter, sum(out_ARNT.iter_sub)/out_ARNT.iter, out_ARNT.nrmG, tsolve_ARNT);
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

    function data = fun_extra(data)
        % store some intermediate variables to save computations
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
        % Euclidean gradient
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
        % Euclidean Hessian at X along U
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
