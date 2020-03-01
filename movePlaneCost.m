function [f,g,e,J] = movePlaneCost(x,imus,w,constraints,loss,r)
%% Initialize
if nargin < 3
    w = ones(3,1);
end
if nargin < 4 || (nargin >= 9 && isempty(constraints))
    constraints = [1 4];
end
if nargin < 5
    loss = @(e) lossFunctions(e,'squared');
end
if nargin < 6
    r = zeros(6,1);
end

% Load data variables from imus struct
acc1 = imus.acc1;
gyr1 = imus.gyr1;
quat1 = imus.quat1;
acc2 = imus.acc2;
gyr2 = imus.gyr2;
quat2 = imus.quat2;
if ~isempty(acc1)
    N = size(acc1,2);
elseif ~isempty(gyr1)
    N = size(gyr1,2);
else
    error('Both acc1 and gyr1 cannot be empty.')
end

if isfield(imus,'gyr1_diff') && isfield(imus,'gyr2_diff')
    gyr1_diff = imus.gyr1_diff;
    gyr2_diff = imus.gyr2_diff;
else
    gyr1_diff = zeros(3,N);
    gyr2_diff = zeros(3,N);
end

Nc = length(constraints);
e = zeros(N*Nc,1); % Residuals
g = zeros(4,1); % Gradient
f = 0; % Cost function value
J = zeros(N*Nc,4); % Jacobian

% Accelerometer weights
if ~isfield(imus,'wa')
    wa = ones(N,1);
else
    wa = imus.wa;
end
    
% Gyroscope weights
if ~isfield(imus,'wg')
    wg = ones(N,1);
else
    wg = imus.wg;
end

% Current estimated normal vectors
x1 = x(1:2,1);
x2  = x(3:4,1);
n1 = [cos(x1(1))*cos(x1(2)) cos(x1(1))*sin(x1(2)) sin(x1(1))]';
n2 = [cos(x2(1))*cos(x2(2)) cos(x2(1))*sin(x2(2)) sin(x2(1))]';

% Partial derivatives of normal vectors n w.r.t. spherical coordinates x
dn1dx1 = [-sin(x1(1))*cos(x1(2)) -sin(x1(1))*sin(x1(2)) cos(x1(1));
          -cos(x1(1))*sin(x1(2))  cos(x1(1))*cos(x1(2)) 0];
dn2dx2 = [-sin(x2(1))*cos(x2(2)) -sin(x2(1))*sin(x2(2)) cos(x2(1));
          -cos(x2(1))*sin(x2(2))  cos(x2(1))*cos(x2(2)) 0];

%% Evaluate cost function and Jacobian
g1 = zeros(3,1);
g2 = zeros(3,1);
a1 = zeros(3,1);
a2 = zeros(3,1);
R1 = zeros(3,3);
R2 = zeros(3,3);
for k = 1:N
        %% Current measurements
        if ~isempty(gyr1)
            g1 = gyr1(:,k);
        end
        if ~isempty(gyr2)
            g2 = gyr2(:,k);
        end
        if ~isempty(acc1)
            a1 = acc1(:,k);
        end
        if ~isempty(acc2)
            a2 = acc2(:,k);
        end
        if ~isempty(quat1)
            R2 = quat2mat(quat2(:,k));
        end
        if ~isempty(quat2)
            R1 = quat2mat(quat1(:,k));
        end
        g1d = gyr1_diff(:,k);
        g2d = gyr2_diff(:,k);
        
        %% Compute constraints
        jk = k;
        for j = 1:length(constraints)
            switch constraints(j)
                case 1
                    % A1) Accelerations along normal axes of equal magnitude
                    if ~exist('r','var')
                        ea1 = dot(a1,n1) - dot(a2,n2);
                        dea1dn1 = w(j)*wa(k)*a1;
                        dea1dn2 = -w(j)*wa(k)*a2;
                        J(jk,:) = [(dn1dx1*dea1dn1)' (dn2dx2*dea1dn2)'];
                    else
                        K1 = omegaMatrix(g1,g1d);
                        K2 = omegaMatrix(g2,g2d);
                        r1 = r(1:3,1);
                        r2 = r(4:6,1);
                        ea1 = dot(a1,n1) - dot(a2,n2) - dot(K1*r1,n1) + dot(K2*r2,n2);
                        dea1dn1 = w(j)*wa(k)*(a1-K1*r1);
                        dea1dn2 = -w(j)*wa(k)*(a2-K2*r2);
                        J(jk,:) = [(dn1dx1*dea1dn1)' (dn2dx2*dea1dn2)'];
                    end
                    e(jk) = w(j)*wa(k)*ea1;
                    [l,dlde] = loss(e(jk));
                    f = f + l;
                    g = g + dlde*J(jk,:)';
                    jk = jk + N;
                case 2
                    % A2) Accelerations along normal axes should be zero 
                    % A3) No acceleration outside the plane
                    ea1 = dot(a1,n1);
                    ea2 = dot(a2,n2);
                    dea1dn1 = w(j)*a1;
                    dea2dn2 = w(j)*a2;
                    J(jk,:) = [(dn1dx1*dea1dn1)' zeros(1,2)];
                    e(jk) = w(j)*ea1;
                    [l,dlde] = loss(e(jk));
                    f = f + l;
                    g = g + dlde*J(jk,:)';
                    jk = jk + N;
                    J(jk,:) = [zeros(1,2) (dn2dx2*dea2dn2)'];
                    e(jk) = w(j)*ea2;
                    [l,dlde] = loss(e(jk));
                    f = f + l;
                    g = g + dlde*J(jk,:)';
                    jk = jk + N;
                case 3
                    % A4) Projections of accelerations into the plane should have
                    % the same magnitude as the measurements themselves
                    % A3) No acceleration outside the plane
                    na1 = norm(cross(a1,n1),2);
                    na2 = norm(cross(a2,n2),2);
                    ea1 = na1 - norm(a1,2); % Residual for accelerometer 1
                    ea2 = na2 - norm(a2,2); % Residual for acceleroemter 2
                    if na1 == 0
                        dea1dn1 = zeros(3,1);
                    else
                        dea1dn1 = w(j)*cross(cross(a1,n1),a1)/na1;
                    end
                    if na2 == 0
                        dea2dn2 = zeros(3,1);
                    else
                        dea2dn2 = w(j)*cross(cross(a2,n2),a2)/na2;
                    end
                    J(jk,:) = [(dn1dx1*dea1dn1)' zeros(1,2)];
                    e(jk) = w(j)*ea1;
                    [l,dlde] = loss(e(jk));
                    f = f + l;
                    g = g + dlde*J(jk,:)';
                    jk = jk + N;
                    J(jk,:) = [zeros(1,2) (dn2dx2*dea2dn2)'];
                    e(jk) = w(j)*ea2;
                    [l,dlde] = loss(e(jk));
                    f = f + l;
                    g = g + dlde*J(jk,:)';
                    jk = jk + N;
                case 4
                    % A5) Projections of the angular velocities into the
                    %     plane have equal magnitude
                    ng1 = norm(cross(g1,n1),2);
                    ng2 = norm(cross(g2,n2),2);
                    eg = ng1 - ng2;
                    if ng1 == 0 || ng2 == 0
                        degdn1 = zeros(3,1);
                        degdn2 = zeros(3,1);
                    else
                        degdn1 = w(j)*wg(k)*cross(cross(g1,n1),g1)/ng1;
                        degdn2 = -w(j)*wg(k)*cross(cross(g2,n2),g2)/ng2;
                    end
                    J(jk,:) = [(dn1dx1*degdn1)' (dn2dx2*degdn2)'];
                    e(jk) = w(j)*wg(k)*eg;
                    [l,dlde] = loss(e(jk));
                    f = f + l;
                    g = g + dlde*J(jk,:)';
                    jk = jk + N;
                case 5
                    % A6) Planes are parallell in the global frame
                    nr = norm(cross(R1*n1,R2*n2),2);
                    er = nr;
                    if er == 0
                        derdn1 = zeros(3,1);
                        derdn2 = zeros(3,1);
                    else
                        derdn1 = w(j)*(crossMat(R2*n2)'*R1)'*cross(R1*n1,R2*n2)/nr;
                        derdn2 = w(j)*(crossMat(R1*n1)*R2)'*cross(R1*n1,R2*n2)/nr;
                    end
                    J(jk,:) = [(dn1dx1*derdn1)' (dn2dx2*derdn2)'];
                    e(jk) = w(j)*er;
                    [l,dlde] = loss(e(jk));
                    f = f + l;
                    g = g + dlde*J(jk,:)';
                    jk = jk + N;
            end
        end    
end
% f = e;