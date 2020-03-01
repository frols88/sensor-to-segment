function [xhat,optimVars] = jointPosIdent(imu1,imu2,settings,jointAxis)
%% Plane of movement identification
% DESCRIPTION:
% Identify the joint center position
    
%% Initialzie
if ~isfield(imu1,'acc') && ~isfield(imu1,'gyr')
    error('Both acc and gyr are missing from first IMU.')
end
    
% Use default settings if no settings struct is provided
% loss = @(e) lossFunctions(e,'squared');
loss = @(e) lossFunctions(e,'squared');
optOptions = optimOptions(); % Optimization options

if isfield(settings,'dimension')
    if settings.dimension == 2
        dim = 2;
    elseif settings.dimension == 3
        dim = 3;
    else
        warning('Dimension not equal to 2 or 3, defaulting to 3.')
        dim = 3;
    end
end
r0 = randn(2*dim,1);

if nargin > 2
    if isfield(settings,'optOptions')
        optOptions = settings.optOptions;
    end
    if isfield(settings,'r0')
        r0 = settings.r0;
    end
end

%% Optimization
r0 = randn(6,1); % REMOVE LATER
% Define cost function
if dim == 1 % Remember to change back to 2 and 3 below!
    [x(1:3,1),y(1:3,1)] = jointAxisBasisVectors(jointAxis(1:3));
    [x(4:6,1),y(4:6,1)] = jointAxisBasisVectors(jointAxis(4:6));
    costFunc = @(r) jointPosCost(r,imu1,imu2,loss,x,y);
elseif dim == 2
    costFunc = @(r) jointPosCost(r,imu1,imu2,loss);
end

[xhat,optimVars] = optimGaussNewton(r0,costFunc,optOptions);
% [xhat,xtraj] = optimBFGS(r0,costFunc,optOptions);

xhat(1:3) = xhat(1:3) - jointAxis(1:3)'*xhat(1:3)*jointAxis(1:3); 
xhat(4:6) = xhat(4:6) - jointAxis(4:6)'*xhat(4:6)*jointAxis(4:6);

if dim == 1
    xhat = [xhat; x; y]; % Append basis vectors for the 2d estimates
end
