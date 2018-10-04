function [R,acc,gyr,mag,quat] = imuPlaneProject(acc,gyr,mag,quat,zp,tp,time)
% DESCRIPTION: Project IMU measurements into movement planes
%
% INPUT:
% acc       - Accelerometer measurements, size [3,N] or empty
% gyr       - Gyroscope measurements, size [3,N] or empty
% mag       - Magnetometer measurements, size [3,N] or empty
% quat      - Quaternion measurements, size [3,N] or empty
% zp        - Vector with normal vectors of the movement planes, size [3,Np]
% tp        - Vector with the samples/times related to the normal vectors. 
%               Example: Measurements closest in time to tp(i) will be
%               projected into the zp(i) plane. Size [1,Np]
% time      - (Optional) Use only if tp corresponds to sample times in
%               seconds
%
% OUTPUT:
% R         - Rotation matrices corresponding to the plane projections, 
%               size [3,3,Np]
% acc       - Measurement vectors projected into the planes defined by the
% gyr           normal vectors in zp. Dimensions 1 and 2 are inside the
% mag           planes and dimension 3 is aligned with the normal axes.
% quat          Quaternions relate the new projected sensor frames to 
%               the global frame.
% 
N = size(acc,2);
Np = length(tp);
R = zeros(3,3,Np);
for k = 1:Np
    % Find valid xy-axes in the movement plane
    if zp(3,k) ~= 0
        if k == 1
            xp = randn(3,1);
        end
        xp(3) = (-zp(1,k)*xp(1)-zp(2,k)*xp(2))/zp(3,k);
    else
        xp = [0 0 1]';
    end
    xp = xp/norm(xp);
    yp = cross(xp,zp(:,k));
    R(:,:,k) = [xp yp zp(:,k)]';
end

for k = 1:N
    if nargin < 7
        j = find(abs(tp-k) == min(abs(tp-k)));
    else
        j = find(abs(tp-time(k)) == min(abs(tp - time(k))));
    end
    Rj = reshape(R(:,:,j),[3 3]);
    acc(:,k) = Rj*acc(:,k);
    gyr(:,k) = Rj*gyr(:,k);
    mag(:,k) = Rj*mag(:,k);
    quat(:,k) = mat2quat(Rj*quat2mat(quat(:,k)));
end