function [X,y] = CorNewton3(G,b,I,J,tau)
%%%%%%%%% This code is designed to solve %%%%%%%%%%%%%
%%       min    0.5*<X-G, X-G>
%%         s.t. X_ij = b_k     for (i,j) in (I,J)
%%              X>=tau*I       X is PD
%%
%%%%%%%%%%%    Based on the algorithm  in  %%%%%%%%%%%%%%
%%%%  "A Quadratically Convergent Newton Method for  %%%%
%%%%   Computing the Nearest Correlation Matrix"     %%%%
%%%%%%%%%%    By Houduo Qi and Defeng Sun    %%%%%%%%%%%%
%%%   SIAM J. Matrix Anal. Appl. 28 (2006) 360--385.  %%%

%%%%%%%%%%%%%%%%%%%% Parameters 
%   Input
%   G       the given symmetric correlation matrix
%   b       the right hand side of equality constraints
%   I       row indices of the fixed elements
%   J       column indices of the fixed elements
%
%   Output
%   X         the optimal primal solution
%   y         the optimal dual solution 
%%%%%%%%%%%%%%%%%%%
%%%%%%   Send your comments and suggestions to        %%%%%%
%%%%%%   hdqi@soton.ac.uk  or matsundf@nus.edu.sg     %%%%%%
%%%%%%   Warning: Accuracy may not be guaranteed!     %%%%%%
%%% Last modified by Yan Gao and Defeng Sun on  September 8, 2009   


t0 = clock;
n = length(G);
k = length(b);

%fprintf('\n ******************************************************** \n')
%fprintf( '          The Semismooth Newton-CG Method                ')
%fprintf('\n ******************************************************** \n')
%fprintf('\n The information of this problem is as follows: \n')
%fprintf(' Dim. of    sdp      constr  = %d \n',n)
%fprintf(' Num. of equality    constr  = %d \n',k)

G  = (G+G')/2;    % make G symmetric
b0 = b;

if nargin==5
    G        = G-tau*eye(n);   % reset G
    Ind      = find(I==J);      % reset the diagonal part of b0   
    b0(Ind)  = b0(Ind)-tau;   
end

%%% set parameters 
Iter_Whole = 200;
Iter_inner = 20;      % maximum num of Line Search in Newton method
maxit      = 200;     % maximum num of iterations in PCG

error_tol = 1.0e-6;    % termination tolerance
tol       = 1.0e-2;    % relative accuracy for CGs
sigma     = 1.0e-4;    % tolerance in the line search of the Newton method

k1      = 0;
f_eval  = 0;
iter_cg = 0;

prec_time = 0;
pcg_time  = 0;
eig_time  = 0;

num = 5;
f_hist = zeros(num,1);

%initial point
y  = zeros(k,1);
x0 = y;

X= zeros(n,n);
for i=1:k
X(I(i),J(i)) = y(i);
end
X = 0.5*(X + X');
X = G + X;
X = (X + X')/2;

t1       = clock;
[P,D]    = eig(X);
eig_time = eig_time + etime(clock,t1);
lambda   = diag(D);
P        = real(P);
lambda   = real(lambda);
if lambda(1) < lambda(n)  
    lambda = lambda(n:-1:1);
    P = P(:,n:-1:1);
end

[f0,Fy] = gradient(y,I,J,lambda,P,X,b0);
f_eval  = f_eval + 1;
f       = f0;
b       = b0-Fy;
norm_b  = norm(b);

f_hist(1) = f;

Omega12 = omega_mat(lambda);

val_G     = sum(sum(G.*G))/2;
Initial_f = val_G-f0;
%fprintf('\n Initial Dual Objective Function value  = %d \n', Initial_f)

tt = etime(clock,  t0);
[hh,mm,ss] = time(tt);

%fprintf('\n   Iter.   No. of CGs     Step length      Norm of gradient     func. value      time_used ')
%fprintf('\n    %d         %2.0d            %3.2e              %3.2e       %10.9e      %d:%d:%d ',0,str2num('-'),str2num('-'),norm_b,f, hh,mm,ss)

while (norm_b>error_tol && k1<Iter_Whole)
    
    t2 = clock;
    c  = precond_matrix(I,J,Omega12,P);       
    prec_time = prec_time + etime(clock, t2);

    t3 = clock;
    [d,flag,relres,iterk] = pre_cg(b,I,J,tol,maxit,Omega12,P,c);
    pcg_time = pcg_time + etime(clock, t3);
    iter_cg = iter_cg + iterk;
   
    slope = (Fy-b0)'*d; 

    y = x0 + d;   
    
    X = zeros(n,n);
    for i = 1:k
        X(I(i),J(i)) = y(i);
    end
    X = 0.5*(X + X');
    X = G + X;
    X = (X + X')/2;

    t1       = clock;
    [P,D]    = eig(X);
    eig_time = eig_time + etime(clock,t1);
    lambda   = diag(D);
    P        = real(P);
    lambda   = real(lambda);
    if lambda(1) < lambda(n)
        lambda = lambda(n:-1:1);
        P = P(:,n:-1:1);
    end
    
    [f,Fy] = gradient(y,I,J,lambda,P,X,b0);

    k_inner=0;
    while( k_inner <= Iter_inner && f > f0 + sigma*0.5^k_inner*slope + 1.0e-6 )        
        y = x0 + 0.5^k_inner*d;    % backtracking       
        
        X = zeros(n,n);
        for i=1:k
            X(I(i),J(i)) = y(i);
        end
        X = 0.5*(X + X');
        X = G + X;
        X = (X + X')/2;

        t1       = clock;
        [P,D]    = eig(X);
        eig_time = eig_time + etime(clock,t1);
        lambda   = diag(D);
        P        = real(P);
        lambda   = real(lambda);
        if lambda(1) < lambda(n)
            lambda = lambda(n:-1:1);
            P = P(:,n:-1:1);
        end

        [f,Fy]   = gradient(y,I,J,lambda,P,X,b0);
         k_inner = k_inner+1;
    end    % End for line search
    
    k1     = k1+1;
    f_eval = f_eval+k_inner+1;
    
    x0     = y;
    f0     = f;
    b      = b0 - Fy;
    norm_b = norm(b);
   
    Omega12 = omega_mat(lambda);
    
    tt = etime(clock, t0);
    [hh,mm,ss] = time(tt); 
    %fprintf('\n   %2.0d         %2.0d           %3.2e          %3.2e       %10.9e      %d:%d:%d ',k1,iterk,0.5^k_inner,norm_b,f, hh,mm,ss)
    
    % slow convergence test
    if  (k1<num)
        f_hist(k1+1) = f;
    else
        for i=1:num-1
            f_hist(i) = f_hist(i+1);
        end
        f_hist(num) = f;
    end  
    if ( k1 >= num-1 && f_hist(1)-f_hist(num) < 1.0e-7 )
        %fprintf('\n Progress is too slow! :( ')
        break;
    end

end   %End of while loop

% Optimal solution X*
Ip = find(lambda>0);
r = length(Ip);
 
if (r==0)
    X = zeros(n,n);
elseif (r==n)
    X = X;
elseif (r<=n/2)
    lambda1 = lambda(Ip);
    
    lambda1 = lambda1.^0.5;
    P1 = P(:,Ip);
    if r >1
        P1 = P1*sparse(diag(lambda1));
        X = P1*P1'; % Optimal solution X*
    else
        X = lambda1^2*P1*P1';
    end
else      
    lambda2 = -lambda(r+1:n);
    lambda2 = lambda2.^0.5;
    P2 = P(:,r+1:n);
    P2 = P2*sparse(diag(lambda2));
    X = X + P2*P2'; % Optimal solution X* 
end
 
X = (X+X')/2;

% optimal primal and dual objective values
Final_f = val_G-f;
val_obj = sum(sum((X-G).*(X-G)))/2;

% convert to original X
X = X + tau*eye(n);

time_used = etime(clock, t0);

%fprintf('\n')
%fprintf('\n ================ Final Information ================= \n');
%fprintf(' Total number of iterations      = %2.0f \n',k1);
%fprintf(' Number of func. evaluations     = %2.0f \n',f_eval)
%fprintf(' Number of CG Iterations         = %2.0f \n',iter_cg)
%fprintf(' Primal objective value          = %d \n', val_obj)
%fprintf(' Dual objective value            = %d \n', Final_f)
%fprintf(' Norm of Gradient                = %3.2e \n', norm_b)
%fprintf(' Rank of X-tau*I             ====== %8.0f \n', r)
%fprintf(' Computing time for preconditioner     = %3.1f \n',prec_time)
%fprintf(' Computing time for CG iterations      = %3.1f \n',pcg_time)
%fprintf(' Computing time for eigen-decom        = %3.1f \n',eig_time)
%fprintf(' Total Computing time (secs)           = %3.1f \n',time_used)
%fprintf(' ====================================================== \n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  End of the main program   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%  **************************************
%%  ******** All Sub-routines  ***********
%%  **************************************

%%% To change the format of time 
function [h,m,s] = time(t)
t = round(t); 
h = floor(t/3600);
m = floor(rem(t,3600)/60);
s = rem(rem(t,60),60);
return
%%% End of time.m



%%% To generate F(y)
function [f,Fy] = gradient(y,I,J,lambda,P,X,b0)
n = length(P);
k = length(y);

const_sparse = 2; % min(5,n/50);

f  = 0.0;
Fy = zeros(k,1);

I1 = find(lambda>1.0e-18);
r  = length(I1);
if (r>0)
    if (r == n)
        f  = lambda'*lambda;
        i=1;
        while (i<=k)
            Fy(i) = X(I(i),J(i));
            i=i+1;
        end                
    elseif (r<=n/2)
        lambda1 = lambda(I1);
        f = lambda1'*lambda1;

        lambda1 = lambda1.^0.5;
        P1 = P(:,I1);
        if r >1
            P1 = P1*sparse(diag(lambda1));
        else
            P1 = lambda1*P1;
        end
        P1T = P1';

        if (k <= const_sparse*n) %% sparse form
            i=1;
            while (i<=k)
                Fy(i) = P1(I(i),:)*P1T(:,J(i));
                i=i+1;
            end
        else %% dense form
            P = P1*P1T;
            i=1;
            while (i<=k)
                Fy(i) = P(I(i),J(i));
                i=i+1;
            end
        end
    else  % n/2<r<n
        lambda2 = -lambda(r+1:n);
        f = lambda'*lambda - lambda2'*lambda2;
        %lambda1 = lambda(I1);
        %f = lambda1'*lambda1;
             
        lambda2 = lambda2.^0.5;
        P2 = P(:, r+1:n);
        P2 = P2*sparse(diag(lambda2));
        P2T = P2';

        if (k<=const_sparse*n) % sparse form
            i=1;
            while (i<=k)
                Fy(i) = X(I(i),J(i)) + P2(I(i),:)*P2T(:,J(i));
                i=i+1;
            end
        else %% dense form
            P = P2*P2T;
            i=1;
            while (i<=k)
                Fy(i) = X(I(i),J(i)) + P(I(i),J(i));
                i=i+1;
            end
        end
    end
end
 f = 0.5*f - b0'*y;
return
%%% End of gradient.m


    
%%% To generate the essential part of the first-order difference of d
function Omega12 = omega_mat(lambda)
% We compute omega only for 1<=|idx|<=n-1
n       = length(lambda);
idx.idp = find(lambda>0);
idx.idm = setdiff([1:n],idx.idp);
r       = length(idx.idp);

if ~isempty(idx.idp)
    if (r == n)
        Omega12 = ones(n,n);
    else
        s  = n-r;
        dp = lambda(1:r);
        dn = lambda(r+1:n);
        
        Omega12 = (dp*ones(1,s))./(abs(dp)*ones(1,s) + ones(r,1)*abs(dn'));
        %Omega12 = max(1e-15,Omega12);
        %Omega = [ones(r) Omega12; Omega12' zeros(s)];
    end
else
    Omega12 = [];
end
return
%%% End of omega_mat.m 



%%%%%%%%%%%%%%%%%%%%%%% PCG method  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This is exactly the algorithm given by Hestenes and Stiefel (1952)
%%% An iterative method to solve A(x)=b  
%%% The symmetric positive definite matrix M is a preconditioner for A.
%%% See Pages 527 and 534 of Golub and va Loan (1996)
function [p,flag,relres,iterk] = pre_cg(b,I,J,tol,maxit,Omega12,P,c)
k1     = length(b);
[dim_n, dim_m] =size(P); 
flag   = 1;
iterk  = 0;
relres = 1000; % give a big value on relres

r    = b;           % initial x0=0 
n2b  = norm(b);     % norm of b
tolb  = max(tol,min(0.1,n2b))*n2b;       % relative tolerance tol*n2b;   % relative tolerance 

if n2b > 1.0e2
    maxit = min(1,maxit);
end
p = zeros(k1,1);

%%% preconditioning
z   = r./c;  % z = M\r; here M =diag(c); if M is not the identity matrix 
rz1 = r'*z; 
rz2 = 1; 
d   = z;

%%% CG iteration
for k=1:maxit
   if (k>1)
       beta = rz1/rz2;
       d    = z + beta*d;
   end
   
   w = Jacobian_matrix(d,I,J,Omega12,P); % W =A(d)
   
  if k1 > dim_n      %% if there are more constraints than n
      w = w + 1.0e-2*min(1.0, 0.1*n2b)*d;  %% perturb it to avoid numerical singularity
  end
   
 
  
   denom  = d'*w;
   iterk  = k;
   relres = norm(r)/n2b;        %relative residue=norm(r)/norm(b)
   
   if denom <= 0 
       p = d/norm(d); % d is not a descent direction
       break;   % exit
   else
       alpha = rz1/denom;
       p     = p + alpha*d;
       r     = r - alpha*w;
   end
   
   z = r./c; %  z = M\r; here M =diag(c); if M is not the identity matrix   
   if (norm(r) <= tolb)   % Exit if Hp=b solved within the relative tolerance
       iterk  = k;
       relres = norm(r)/n2b;          %relative residue =norm(r)/norm(b)
       flag   = 0;
       break;
   end
   rz2 = rz1;
   rz1 = r'*z;
end
return
%%% End of pre_cg.m

 


%%% To generate the Jacobian product with x: F'(y)(x)
function Ax = Jacobian_matrix(x,I,J,Omega12,P)
n      = length(P);
k      = length(x);
[r,s]  = size(Omega12); 

if (r==0)
    Ax = 1.0e-10*x;
elseif (r==n)
    Ax = (1 + 1.0e-10)*x;
else
    Ax = zeros(k,1);
    P1 = P(:,1:r);
    P2 = P(:,r+1:n);
    
    Z = zeros(n,n);
    for i = 1:k
       Z(I(i),J(i)) = x(i);
    end
    Z = 0.5*(Z + Z');
  
    const_sparse = 2;  % min(5,n/50); 
    if (k<=const_sparse*n)
        % sparse form
        if (r<n/2)
            %H = (Omega.*(P'*sparse(Z)*P))*P';
            H1 = P1'*sparse(Z);
            Omega12 = Omega12.*(H1*P2);
            H = [(H1*P1)*P1' + Omega12*P2'; Omega12'*P1'];
           
            i=1;
            while (i<=k)
                Ax(i) = P(I(i),:)*H(:,J(i));
                Ax(i) = Ax(i) + 1.0e-10*x(i);    %add a small perturbation
                i=i+1;
            end
        else % if r>=n/2, use a complementary formula.
            %H = ((E-Omega).*(P'*Z*P))*P';               
            H2 = P2'*sparse(Z);
            Omega12 = ones(r,s)- Omega12;
            Omega12 = Omega12.*((H2*P1)');
            H = [Omega12*P2'; Omega12'*P1' + (H2*P2)*P2'];
           
            i=1;
            while (i<=k)
                %%% AA^* is not the identity matrix
                if (I(i)==J(i))
                    Ax(i) = x(i) - P(I(i),:)*H(:,J(i));
                else
                    Ax(i) = x(i)/2 - P(I(i),:)*H(:,J(i));
                end
                Ax(i) = Ax(i) + 1.0e-10*x(i);   
                i=i+1;
            end
        end
       
    else %dense form
        %Z = full(Z); to use the full form
        % dense form
        if (r<n/2) 
            %H = P*(Omega.*(P'*Z*P))*P';            
            H1 = P1'*Z;
            Omega12 = Omega12.*(H1*P2);            
            H = P1*((H1*P1)*P1'+ 2.0*Omega12*P2');            
            H = (H + H')/2; 
            
            i = 1;
            while (i<=k)
               Ax(i) = H(I(i),J(i));
               Ax(i) = Ax(i) + 1.0e-10*x(i);
               i = i+1;
            end    
        else % if r>=n/2, use a complementary formula.
            %H = - P*( (E-Omega).*(P'*Z*P) )*P';           
            H2 = P2'*Z;
            Omega12 = ones(r,s)-Omega12;
            Omega12 = Omega12.*(H2*P1)';
            H = P2*( 2.0*(Omega12'*P1') + (H2*P2)*P2');            
            H = (H + H')/2;
            H = Z - H;
             
            i = 1;
            while (i<=k)    %%% AA^* is not the identity matrix
                Ax(i) = H(I(i),J(i));
                Ax(i) = Ax(i) + 1.0e-10*x(i);
                i = i+1;
            end
        end            
    end    
end
return
%%% End of Jacobian_matrix.m  



%%% To generate the (approximate) diagonal preconditioner
function c = precond_matrix(I,J,Omega12,P) 
n     = length(P);
k     = length(I);
[r,s] = size(Omega12);

c = ones(k,1);

H = P';
H = H.*H;
const_prec = 1;
if (r<n)
    if (r>0)        
        if (k<=const_prec*n)     % compute the exact diagonal preconditioner
            
            Ind = find(I~=J);
            k1  = length(Ind);
           if (k1>0)
                H1 = zeros(n,k1);
                for i=1:k1
                    H1(:,i) = P(I(Ind(i)),:)'.*P(J(Ind(i)),:)';
                end
            end
            
            if (r<n/2)                  
                H12  = H(1:r,:)'*Omega12;
                if(k1>0)
                    H12_1 = H1(1:r,:)'*Omega12;
                end
                
                d = ones(r,1);
                
                j=0;
                for i=1:k                   
                    if (I(i)==J(i))
                        c(i) = sum(H(1:r,I(i)))*(d'*H(1:r,J(i)));
                        c(i) = c(i) + 2.0*(H12(I(i),:)*H(r+1:n,J(i)));
                    else 
                        j=j+1;
                        c(i) = sum(H(1:r,I(i)))*(d'*H(1:r,J(i)));
                        c(i) = c(i) + 2.0*(H12(I(i),:)*H(r+1:n,J(i)));
                        c(i) = c(i) + sum(H1(1:r,j))*(d'*H1(1:r,j));
                        c(i) = c(i) + 2.0*(H12_1(j,:)*H1(r+1:n,j));
                        c(i) = 0.5*c(i);
                    end
                    if c(i) < 1.0e-8
                        c(i) = 1.0e-8;
                    end                      
                end   
                                
            else  % if r>=n/2, use a complementary formula
                Omega12 = ones(r,s)-Omega12;
                H12  = Omega12*H(r+1:n,:);
                if(k1>0)
                    H12_1 = Omega12*H1(r+1:n,:);
                end
                
                d =  ones(s,1);
                dd = ones(n,1);

                j=0;
                for i=1:k
                    if (I(i)==J(i))
                        c(i) = sum(H(r+1:n,I(i)))*(d'*H(r+1:n,J(i)));
                        c(i) = c(i) + 2.0*(H(1:r,I(i))'*H12(:,J(i)));
                        alpha = sum(H(:,I(i)));
                        c(i) = alpha*(H(:,J(i))'*dd) - c(i);
                    else
                        j=j+1;
                        c(i) = sum(H(r+1:n,I(i)))*(d'*H(r+1:n,J(i)));
                        c(i) = c(i) + 2.0*(H(1:r,I(i))'*H12(:,J(i)));
                        alpha = sum(H(:,I(i)));
                        c(i) = alpha*(H(:,J(i))'*dd) - c(i);

                        tmp = sum(H1(r+1:n,j))*(d'*H1(r+1:n,j));
                        tmp = tmp + 2.0*(H1(1:r,j)'*H12_1(:,j));
                        alpha = sum(H1(:,j));
                        tmp = alpha*(H1(:,j)'*dd) - tmp;
                        
                        c(i) = (tmp + c(i))/2;
                    end                    
                    if c(i) < 1.0e-8
                        c(i) = 1.0e-8;
                    end
                end                
            end
                                   
            
        else  % approximate the diagonal preconditioner
            HH1 = H(1:r,:);
            HH2 = H(r+1:n,:);

            if (r<n/2)
                H0 = HH1'*(Omega12*HH2);
                tmp = sum(HH1);
                H0 = H0 + H0'+ tmp'*tmp;
            else
                Omega12 = ones(r,s) - Omega12;
                H0 = HH2'*((Omega12)'*HH1);
                tmp  = sum(HH2);
                H0 = H0 + H0' + tmp'*tmp;
                tmp = sum(H);
                H0 = tmp'*tmp - H0;
            end

            i=1;
            while (i<=k)
                if (I(i)==J(i))
                    c(i) = H0(I(i),J(i));
                else
                    c(i) = 0.5*H0(I(i),J(i));
                end
                if  c(i) < 1.0e-8
                    c(i) = 1.0e-8;
                end
                i = i+1;
            end
        end        
    end  %End of second if
    
else % if r=n
    tmp = sum(H);
    H0  = tmp'*tmp;
    
    i=1;
    while (i<=k)
        if (I(i)==J(i))
            c(i) = H0(I(i),J(i));
        else
            c(i) = 0.5*H0(I(i),J(i));
        end
        if (c(i)<1.0e-8)
            c(i) = 1.0e-8;
        end
        i = i+1;
    end
end  %End of the first if
return
%%% End of precond_matrix.m 







