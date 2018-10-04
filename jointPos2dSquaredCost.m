function [f,J,e] = jointPos2dSquaredCost(x,acc1,acc2,gyr1,gyr2,gyr1_diff,gyr2_diff)
%% Initialize
if ~isempty(acc1) && size(acc1,1) > 2
    acc1 = acc1(1:2,:);
end
if ~isempty(acc2) && size(acc2,1) > 2
    acc2 = acc2(1:2,:);
end
if ~isempty(gyr1) && size(gyr1,1) > 1
    gyr1 = gyr1(end,:);
    gyr1_diff = gyr1_diff(end,:);
end
if ~isempty(gyr2) && size(gyr2,1) > 1
    gyr2 = gyr2(end,:);
    gyr2_diff = gyr2_diff(end,:);
end

if ~isempty(acc1)
    N = size(acc1,2);
elseif ~isempty(gyr1)
    N = size(gyr1,2);
else
    error('Both acc1 and gyr1 cannot be empty.')
end

J = zeros(N,4);
e = zeros(N,1);

%% Compute cost function and Jacobian
a1 = zeros(2,1);
a2 = zeros(2,1);
g1 = 0;
g2 = 0;
gd1 = 0;
gd2 = 0;
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
    K1 = omegaMatrix2d(g1,gd1);
    K2 = omegaMatrix2d(g2,gd2);
    e1 = a1-K1*x(1:2);
    e2 = a2-K2*x(3:4);
    norm1 = norm(e1,2);
    norm2 = norm(e2,2);
    e(k) = norm1 - norm2;
    if norm(e1) > 0
        J(k,1:2) = -K1'*e1/norm(e1);
    end
    if norm(e2) > 0
        J(k,3:4) = K2'*e2/norm(e2);
    end
end
f = sum(e.^2);