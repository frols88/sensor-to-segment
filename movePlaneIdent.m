function [nhat,that,etot,xtraj] = movePlaneIdent(acc1,acc2,gyr1,gyr2,quat1,quat2,settings)
%% Plane of movement identification
% DESCRIPTION:
% Identify the normal vectors to the planes of movement of two IMU:s

% INPUT:
% acc1,2    - Accelerometer measurements of IMU 1 and 2, size [3,N]
% gyr1,2    - Gyroscope measurements of IMU 1 and 2, size [3,N]
% quat1,2   - Rotation quaternions from global frame to sensor frame
%                1 and 2, size [4,N]
% settings  - Struct with settings parameters, see below.

% OUTPUT:
% nhat      - Estimated normal vectors in leg and torso sensor frames,
%             one estimate per window [6,Nwin]
% that      - Sample indices in the middle of each window [1,Nwin]
% etot      - Residuals
% xtraj     - Trajectories from the initial to the estimated normal
%               vectors, in spherical coordinates
%
% SETTINGS: The following settings can be included as fields in the
%               settings struct:
% winSize       - Window size in number of samples used in a sliding 
%                   window optimization scheme
% overlap       - Overlap between windows between, 0 < overlap < 1
% w             - Weights for the constraints in the cost function
% constraints   - Vector with integer numbers that decide which constraints
%                   in movePlaneCost.m that become active, same length as w
% optOptions    - Optimization options, see optimOptions.m
    
%% Initialzie
if ~isempty(acc1)
    N = size(acc1,2);
elseif ~isempty(gyr1)
    N = size(gyr1,2);
else
    error('Both acc1 or gyr1 cannot be empty.')
end

% Use default settings if no settings struct is provided
winSize = N; % Window size
overlap = 0; % Overlap ratio (Between 0 and 1)
w = [1 1 1]; % Kinematic constraints weights
constraints = [1 4]; % Active constraints
optOptions = optimOptions(); % Optimization options
if nargin > 6
    if isfield(settings,'winSize')
        winSize = settings.winSize;
    end
    if isfield(settings,'overlap')
        overlap = settings.overlap;
    end
    if isfield(settings,'w')
        w = settings.w;
    end
    if isfield(settings,'constraints')
        constraints = settings.constraints;
    end
    if isfield(settings,'optOptions')
        optOptions = settings.optimOptions();
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
x0 = -pi + 2*pi*rand(4,1); % Initialize as uniformly random unit vectors
nhat = zeros(6,Nwin);
that = zeros(1,Nwin);
xtraj = zeros(4,optOptions.maxSteps+1,Nwin);
acc1j = [];
gyr1j = [];
quat1j = [];
acc2j = [];
gyr2j = [];
quat2j = [];
for j = 1:Nwin
    % Select data of the j:th window
    jj = (j-1)*(winSize-overlap)+window;
    if j == Nwin
        jj = N-winSize+1:N;
    end
    if ~isempty(acc1)
        acc1j = acc1(:,jj);
    end
    if ~isempty(acc2)
        acc2j = acc2(:,jj);
    end
    if ~isempty(gyr1)
        gyr1j = gyr1(:,jj);
    end
    if ~isempty(gyr2)
        gyr2j = gyr2(:,jj);
    end
    if ~isempty(quat1)
        quat1j = quat1(:,jj);
    end
    if ~isempty(quat2)
        quat2j = quat2(:,jj);
    end  
    
    % Optimize cost function
    disp(['Identifying movement plane for samples ',num2str(jj(1)),':',num2str(jj(end)),'.'])
    disp(['Active constraints: ',num2str(constraints)])
    costFunc = @(x) movePlaneCost(x,acc1j,acc2j,gyr1j,gyr2j,quat1j,quat2j,w,constraints);
    [x,xtraj(:,:,j)] = optimGaussNewton(x0,costFunc,optOptions);
    [~,~,e] = costFunc(x); % Compute residuals for current x

    % Save results
    n = [[cos(x(1))*cos(x(2)) cos(x(1))*sin(x(2)) sin(x(1))]'; ...
            [cos(x(3))*cos(x(4)) cos(x(3))*sin(x(4)) sin(x(3))]']; % Convert from spherical coordinates to unit vectors
    nhat(:,j) = n(:,end);
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