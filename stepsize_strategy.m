clear
close all

lambda_F = 0.5;
C = 1.000001;
L = 1;

max_iter = 10;
gamma = zeros(max_iter+1,1);
gamma(1) = 1/L;

lower_bound = zeros(max_iter,1);
sum_gamma = zeros(max_iter+1,1);
sum_gamma(1) = gamma(1);

k = 1;
while k < max_iter+1
    k = k+1;
    gamma(k) = min(gamma(k-1),max(C/(k-1),gamma(k-1)/(1+lambda_F)));
    lower_bound(k-1) = gamma(k-1)/(1+lambda_F);
    sum_gamma(k) = sum_gamma(k-1)+gamma(k);
end


scatter(0:max_iter,gamma,200,'xk','DisplayName','Step size \gamma_k')
hold on
scatter(1:max_iter,lower_bound,50,'^b','filled','DisplayName','\gamma_{k-1}/(1+\lambda_F)')
scatter(0:max_iter,C./(0:max_iter),50,'square','filled','Color',"#D95319",'DisplayName','C/k')
axis([-0.3,max_iter+.1,0,2])
legend

%% sums of parameters
weird_sum = zeros(max_iter+1,1);

for K = 1 : max_iter+1
    weird_sum(K) = 0;
    for k = 0 : K-1
        weird_sum(K) = weird_sum(K) + gamma(k+1)*prod(1-lambda_F*gamma(k+2:K));
    end
end

M1 = 0;
N = find(gamma(2:end)==C./(1:max_iter)',1)-1;
for k = 0 : N
    M1 = M1 + gamma(k+1)*prod(1-lambda_F*gamma(k+2:end));
end
fprintf('Predicted upper bound for AK: %f\n',M1+C)