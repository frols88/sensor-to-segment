function [x,xtraj] = optimGaussNewton(x,costFunc,options)
%% Initialize
% Gauss-Newton settings
if nargin < 3
    options = optimOptions();
end
tol = options.tol;
maxSteps = options.maxSteps;
alpha = options.alpha;
beta = options.beta;
f_prev = 0;
diff = tol+1;
step = 1;
xtraj = zeros(size(x,1),maxSteps+1);
xtraj(:,step) = x;

%% Gauss-Newton optimization
while step < maxSteps && diff > tol
    % Evaluate cost function, Jacobian and residual
    [f,J,e] = costFunc(x);
    
    % Backtracking line search
    len = 1; % Initial step size
    dx = pinv(J)*e; % Search direction
    [f_next,~,~] = costFunc(x - len*dx);
    while f_next > f + alpha*len*J*dx
        len = beta*len;
        [f_next,~,~] = costFunc(x - len*dx);
    end
    
    % Update
    x = x - len*dx;
    step = step+1;
    xtraj(:,step) = x;
    if step > 2
        diff = norm(f_prev-f_next);
    end
    f_prev = f_next;
    
    % Print cost function value
    disp(['Gauss-Newton. Step ',num2str(step-1),'. f = ',num2str(f_next),'.'])
    if step > maxSteps
        disp('Gauss-Newton. Maximum iterations reached.')
    elseif diff <= tol
        disp('Gauss-Newton. Cost function update less than tolerance.')
    end
end
xtraj(:,step:end) = repmat(NaN*ones(size(x,1),1),[1 size(xtraj(:,step:end),2)]);
disp(['Gauss-Newton. Stopped after ',num2str(step),' iterations.'])