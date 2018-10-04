function [rhat,that,etot,rtraj] = jointPosIdent(acc1,acc2,gyr1,gyr2,time,settings)
%% Identification of joint position
% DESCRIPTION: Identify the position of a joint with respect to the
%               respective sensor frames of inertial sensors attached to 
%               rigid body segments adjacent to the joint.

% INPUT:
% acc1,2    - Accelerometer measurements of IMU 1 and 2, size [3,N]
% gyr1,2    - Gyroscope measurements of IMU 1 and 2, size [3,N]
% time      - Time vector, size [1,N]
% settings  - Struct with settings parameters, see below.

% OUTPUT:
% rhat      - Estimated joint position in the coordinate frames of each
%               IMU.
% that      - Sample indices in the middle of each window, size [1,Nwin].
% etot      - Residuals of the kinematic constraints
% rtraj     - Trajectories from the initial to the estimated position
%               across the optimization procedure

% SETTINGS: The following settings can be included as fields in the
%               settings struct:
% winSize       - Window size in number of samples used in a sliding 
%                   window optimization scheme
% overlap       - Overlap between windows between, 0 < overlap < 1
% method        - String denoting which optimization method that is used:
%                   '3dSquaredCost'     - jointPos3dSquaredCost.m
%                   '3dAbsoluteCost'    - jointPos3dAbsoluteCost.m
%                   '2dSquaredCost'     - jointPos2dSquaredCost.m
% optOptions    - Optimization options  - optimOptions.m
    
%% Initialzie
if ~isempty(acc1)
    N = size(acc1,2);
elseif ~isempty(gyr1)
    N = size(gyr1,2);
else
    error('Both acc1 or gyr1 cannot be empty.')
end

% Differentiate gyro measurements
gyr1_diff = imuDiff(gyr1,time);
gyr2_diff = imuDiff(gyr2,time);
acc1 = acc1(:,2:end-1);
acc2 = acc2(:,2:end-1);
gyr1 = gyr1(:,2:end-1);
gyr2 = gyr2(:,2:end-1);
gyr1_diff = gyr1_diff(:,2:end-1);
gyr2_diff = gyr2_diff(:,2:end-1);
N = N - 2;

% Use default settings if no settings struct is provided
winSize = N; % Window size
overlap = 0; % Overlap ratio (Between 0 and 1)
method = '3dSquaredCost';
optOptions = optimOptions();
if nargin > 5
    if isfield(settings,'winSize')
        winSize = settings.winSize;
    end
    if isfield(settings,'overlap')
        overlap = settings.overlap;
    end
    if isfield(settings,'method')
        method = settings.method;
    end
    if isfield(settings,'optOptions')
        optOptions = settings.optOptions;
    end
end
if winSize > N
    error('Window size cannot be larger than N.')
end
if overlap < 0 || overlap > 1
    error('Overlap has to be in the interval (0,1).')
end

% Sliding window variables
overlap = ceil(overlap*winSize); % Convert overlap ratio to number of samples
window = 1:winSize;
Nwin = floor((N-winSize)/(winSize-overlap))+1;

%% Optimization
x0 = -1 + 2*rand(6,1); % Initialize as uniformly random unit vectors
rhat = zeros(6,Nwin);
that = zeros(1,Nwin);
rtraj = zeros(6,optOptions.maxSteps+1,Nwin);
acc1j = [];
gyr1j = [];
gyr1j_diff = [];
acc2j = [];
gyr2j = [];
gyr2j_diff = [];
for j = 1:Nwin
    % Select data of the j:th window
    jj = (j-1)*(winSize-overlap)+window;
    if j == Nwin
        jj = N-winSize+1:N;
    end
    if ~isempty(acc1) && ~isempty(gyr1) && ~isempty(gyr1_diff)
        acc1j = acc1(:,jj);
        gyr1j = gyr1(:,jj);
        gyr1j_diff = gyr1_diff(:,jj);
    end
    if ~isempty(acc2) && ~isempty(gyr2) && ~isempty(gyr2_diff)
        acc2j = acc2(:,jj);
        gyr2j = gyr2(:,jj);
        gyr2j_diff = gyr2_diff(:,jj);
    end
    
    % Optimize cost function
    disp(['Identifying sensor position for samples ',num2str(jj(1)),':',num2str(jj(end)),'.'])
    disp(['Using method: ',method,'.'])
    if strcmp(method,'3dSquaredCost')
        costFunc = @(x) jointPos3dSquaredCost(x,acc1j,acc2j,gyr1j,gyr2j,gyr1j_diff,gyr2j_diff);
        [x,rtraj] = optimGaussNewton(x0,costFunc,optOptions);
    elseif strcmp(method,'3dAbsoluteCost')
        costFunc = @(x) jointPos3dAbsoluteCost(x,acc1j,acc2j,gyr1j,gyr2j,gyr1j_diff,gyr2j_diff);
        [x,rtraj] = optimGradientDescent(x0,costFunc,optOptions);
    elseif strcmp(method,'2dSquaredCost')
        costFunc = @(x) jointPos2dSquaredCost(x,acc1j,acc2j,gyr1j,gyr2j,gyr1j_diff,gyr2j_diff);
        if j == 1
            x0 = x0(1:4);
            rhat = rhat(1:4,:);
            rtraj = rtraj(1:4,:,:);
        end
        [x,rtraj(:,:,j)] = optimGaussNewton(x0,costFunc,optOptions);
    else
        error(['Undefined method: ',method,'.'])
    end
    [~,~,e] = costFunc(x); % Compute residuals for current x

    % Save results
    rhat(:,j) = x;
    that(:,j) = median(jj);
    if ~exist('etot','var')
        etot = zeros(winSize,round(size(e,1)/winSize),Nwin);
    end
    if j < Nwin
        e = reshape(e,[winSize round(size(e,1)/winSize)]);
    else
        e = reshape(e,[length(jj) round(size(e,1)/length(jj))]);
        e = [e; ones(winSize-length(jj),size(e,2))*NaN];
    end 
    etot(:,:,j) = e;
    x0 = x;
end