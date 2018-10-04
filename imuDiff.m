function diffMeas = imuDiff(meas,time)
% DESCRIPTION: Differentiate measurements using a first order 
%               central difference method.
%
% INPUT:
% meas      - Measurement matrix with time along the 2nd dimension.
% time      - Time vector with each element corresponding to a sample
%               instance in seconds.
%
% OUTPUT:
% diffMeas  - Measurements differentiated w.r.t. time.

N = size(meas,2);
diffMeas = zeros(size(meas));
for k = 2:N-1
    h(1) = time(k)-time(k-1);
    h(2) = time(k+1)-time(k);
    diffMeas(:,k) = (meas(:,k+1)-meas(:,k-1))/(sum(h));
end