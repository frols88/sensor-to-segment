function [wg,wa] = jointAxisWeights(gyr1,acc1,gyr2,acc2,parameters)

if ~isempty(gyr1)
    N = size(gyr1,2);
elseif nargin > 1 && ~isempty(acc1)
    N = size(acc1,2);
end

if nargin < 3 || isempty(gyr2)
    gyr2 = zeros(3,N);
end
if nargin < 4 || isempty(acc2)
    acc2 = zeros(3,N);
end

if nargin > 4 && isfield(parameters,'weightType')
    weightType = parameters.weightType;
else
    weightType = 'equal';
end  

switch weightType
    case 'variance'
        if isfield(parameters,'sg') && isfield(parameters,'sa')
            wg = sqrt(parameters.sa); %parameters.sa/parameters.sg;
            wa = 1/sqrt(parameters.sa); %zeros(N,1);
%             if isfield(parameters,'waPenalty')
%                 waPenalty = parameters.waPenalty;
%             else
%                 waPenalty = 1;
%             end
%             for k = 1:N
%                 gyr1norm = norm(gyr1(:,k));
%                 gyr2norm = norm(gyr2(:,k));
%                 acc1norm = norm(acc1(:,k));
%                 acc2norm = norm(acc2(:,k));
%                 accPenalty = abs(acc1norm-acc2norm);
%                 pseudoVariance = accPenalty^2;
%                 wa(k) = sqrt(1/(1+waPenalty*pseudoVariance));
%             end
        else
            error('Parameters required for specified weight type do not exist.')
        end
    case 'constant'
        if isfield(parameters,'wg') && isfield(parameters,'wa')
            wg = parameters.wg;
            wa = parameters.wa;
        elseif isfield(parameters,'wg')
            wg = parameters.wg;
            wa = 1;
        else
            error('Parameters required for specified weight type do not exist.')
        end
    case 'equal'
        wg = 1;
        wa = 1;
    otherwise
        warning('Specified weight type not found, using default equal weights.')
        wg = 1;
        wa = 1;
end