function [x,xtraj] = optimBFGS(x,costFunc,options)
%% Initialize
% BFGS
if nargin < 3
    options = optimOptions();
end
tol = options.tol;
maxSteps = options.maxSteps;
alpha = options.alpha;
beta = options.beta;
f_prev = 0;
g_prev = 0;
diff = tol+1;
step = 1;
xtraj = zeros(size(x,1),maxSteps+1);
xtraj(:,step) = x;

%% BFGS optimization
while step < maxSteps && diff > tol
    % Evaluate cost function, Jacobian and residual
    [f,g] = costFunc(x);
    
    % Compute approximate inverse Hessian and search direction
    if step > 1
        sk = len*dx;
        yk = g - g_prev;
        Binv = Binv + (sk'*yk + yk'*Binv*yk)*(sk*sk')/((sk'*yk)^2) - (Binv*yk*sk'+sk*yk'*Binv)/(sk'*yk); % Sherman-Morrisson
        dx = -Binv*g;
    else
        N = length(g);
        Binv = eye(N);
        dx = -g;
    end
    
    % Backtracking line search
    len = 1; % Initial step size
    [f_next,~,~] = costFunc(x + len*dx);
    while f_next > f + alpha*len*g'*dx
        len = beta*len;
        [f_next,~,~] = costFunc(x + len*dx);
    end
    
    % Update
    if any(isnan(dx))
       stophere = 1; 
    end
    x = x + len*dx;
    step = step+1;
    xtraj(:,step) = x;
    if step > 2
        diff = norm(f_prev-f_next);
    end
    f_prev = f_next;
    g_prev = g;
    
    % Print cost function value
%     disp(['BFGS. Step ',num2str(step-1),'. f = ',num2str(f_next),'.'])
    if step > maxSteps
%         disp('BFGS. Maximum iterations reached.')
    elseif diff <= tol
%         disp('BFGS. Cost function update less than tolerance.')
    end
end
xtraj(:,step:end) = repmat(NaN*ones(size(x,1),1),[1 size(xtraj(:,step:end),2)]);
% disp(['BFGS. Stopped after ',num2str(step),' iterations.'])