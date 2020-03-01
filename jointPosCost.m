function [f,g,e,J] = jointPosCost(r,imu1,imu2,loss,x,y)
%% Initialzie
% if size(r,1) ~= 4
%     r = r';
% end
% if size(r,1) ~= 4 || size(r,2) ~= 1
%     error('Expecting r to be a 4-dim vector.')
% end
if ~exist('loss','var')
    loss = @(e) lossFunctions(e,'squared');
end

N = size(imu1.acc,2);
e = zeros(N,1); % Residuals
f = 0; % Initialize cost function value

if length(r) > 4
    J = zeros(N,6); % Jacobian
    g = zeros(6,1); % Gradient
    r1 = r(1:3);
    r2 = r(4:6);
    dr1dxy = 1;
    dr2dxy = 1;
elseif exist('x','var') && exist('y','var')
    J = zeros(N,4); % Jacobian
    g = zeros(4,1); % Gradient
    r1 = r(1)*x(1:3,1) + r(2)*y(1:3,1);
    r2 = r(3)*x(4:6,1) + r(4)*y(4:6,1);
    dr1dxy = [x(1:3,1)'; y(1:3,1)'];
    dr2dxy = [x(4:6,1)'; y(4:6,1)'];
else
    error('Basis vectors need to be provided if joint position r is in 2d')
end

%% Evaluate cost function and Jacobian
for k = 1:N
    a1 = imu1.acc(:,k);
    a2 = imu2.acc(:,k);
    g1 = imu1.gyr(:,k);
    g2 = imu2.gyr(:,k);
    g1d = imu1.gyr_diff(:,k);
    g2d = imu2.gyr_diff(:,k);
    K1 = omegaMatrix(g1,g1d);
    K2 = omegaMatrix(g2,g2d);
    norm1 = norm(a1-K1*r1,2);
    norm2 = norm(a2-K2*r2,2);
    e(k) = norm1 - norm2;
    [l,dlde] = loss(e(k));
    if norm1 > 0
        if length(r) > 4
            J(k,1:3) = (dr1dxy*(-K1'*(a1-K1*r1)/norm1))';
        else
            J(k,1:2) = (dr1dxy*(-K1'*(a1-K1*r1)/norm1))';
        end
    end
    if norm2 > 0
        if length(r) > 4
            J(k,4:6) = (dr2dxy*(K2'*(a2-K2*r2)/norm2))';
        else
            J(k,3:4) = (dr2dxy*(K2'*(a2-K2*r2)/norm2))';
        end
    end
    f = f + l;
    g = g + dlde*J(k,:)';
end