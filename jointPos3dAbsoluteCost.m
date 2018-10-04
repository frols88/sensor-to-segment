function [f,g,e] = jointPos3dAbsoluteCost(x,acc1,acc2,gyr1,gyr2,gyr1_diff,gyr2_diff)
%% Initialzie
if size(x,1) ~= 6
    x = x';
end
if size(x,1) ~= 6 || size(x,2) ~= 1
    error('Expecting x to be a 6-dim vector.')
end
if ~isempty(acc1)
    N = size(acc1,2);
elseif ~isempty(acc2)
    N = size(acc2,2);
else
    error('Both acc1 and acc2 cannot be empty.')
end
J = zeros(N,6); % Jacobian
e = zeros(N,1); % Residuals
r1 = x(1:3,1);
r2 = x(4:6,1);

%% Evaluate cost function and Jacobian
g1 = zeros(3,1);
g2 = zeros(3,1);
a1 = zeros(3,1);
a2 = zeros(3,1);
gd1 = zeros(3,1);
gd2 = zeros(3,1);
for k = 1:N
    if ~isempty(acc1) && ~isempty(gyr1) && ~isempty(gyr1_diff)
        a1 = acc1(:,k);
        g1 = gyr1(:,k);
        gd1 = gyr1_diff(:,k);
    end
    if ~isempty(acc2) && ~isempty(gyr2) && ~isempty(gyr2_diff)
        a2 = acc2(:,k);
        g2 = gyr2(:,k);
        gd2 = gyr2_diff(:,k);
    end
    K1 = omegaMatrix(g1,gd1);
    K2 = omegaMatrix(g2,gd2);
    norm1 = norm(a1-K1*r1,2);
    norm2 = norm(a2-K2*r2,2);
    e(k) = norm(norm1 - norm2,1);
    sgn = sign(norm1-norm2);
    if norm1 > 0
        J(k,1:3) = sgn*omegaMatrix(g1,gd1)'*(a1-K1*r1)/norm1;
    end
    if norm2 > 0
        J(k,4:6) = -sgn*omegaMatrix(g2,gd2)'*(a2-K2*r2)/norm2;
    end
end
f = sum(e); % Cost function
g = -sum(J,1)'; % Gradient