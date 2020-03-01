%% Settings
w0 = 50; % Parameter that sets the relative weighting of gyroscope to accelerometer residual, w0 = wg/wa

% Sample selection parameters
useSampleSelection = 0; % Boolean flag to use sample selection
dataSize = 1000; % Maximum number of samples that will be kept after sample selection
winSize = 21; % Window size for computing the average angular rate energy, should be an odd integer
angRateEnergyThreshold = 1; % Theshold for the angular rate energy

% Add utility files to path
addpath([pwd,'\Utility\']);

%% Load data
% Data that needs to be loaded
% acc - 6xN, rows 1:3 from sensor 1, rows 4:6 from sensor 2
% gyr - 6xN, rows 1:3 from sensor 1, rows 4:6 from sensor 2

%% Sample selection
if useSampleSelection
    sampleSelectionVars.dataSize = dataSize;
    sampleSelectionVars.winSize = winSize;
    sampleSelectionVars.angRateEnergyThreshold = angRateEnergyThreshold;
    sampleSelectionVars.deltaGyr = [];
    sampleSelectionVars.gyrSamples = [];
    sampleSelectionVars.accSamples = [];
    sampleSelectionVars.accScore = [];
    sampleSelectionVars.angRateEnergy = [];  
    [gyr,acc,sampleSelectionVars] = jointAxisSampleSelection([],[],gyr,acc,1:length(acc),sampleSelectionVars);
end

%% Identification
settings.x0 = [0 0 0 0]'; % Initial estimate
settings.wa = 1/sqrt(w0); % Accelerometer residual weight
settings.wg = sqrt(w0); % Gyroscope residual weight

imu1 = struct('acc',acc(1:3,:),'gyr',gyr(1:3,:));
imu2 = struct('acc',acc(4:6,:),'gyr',gyr(4:6,:));
[jhat,xhat,optimVarsAxis] = jointAxisIdent(imu1,imu2,settings);