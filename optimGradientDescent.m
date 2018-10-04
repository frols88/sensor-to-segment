function [x,xtraj] = optimGradientDescent(x,costFunc,options)
%% Initialize
% Gradient descent options
if nargin < 3
    options = optimOptions();
end
tol = options.tol;
maxSteps = options.maxSteps;
alpha = options.alpha;
beta = options.beta;
f_prev = 0;
diff = 10;
step = 1;
xtraj = zeros(size(x,1),maxSteps+1);
xtraj(:,step) = x;

%% Gradient descent optimization
while step < maxSteps && diff > tol
    % Evaluate cost function and gradient
    [f,g] = costFunc(x);
    
    % Backtracking line search
    len = 1;
    [f_next,~] = costFunc(x-len*g);
    while f_next > f - alpha*len*norm(g,2)^2
        len = beta*len;
        [f_next,~] = costFunc(x-len*g);
    end
    
    % Update
    x = x - len*g;
    step = step + 1;
    xtraj(:,step) = x;
    if step > 2
        diff = norm(f_prev-f_next);
    end
    f_prev = f_next;
    
    % Print cost function value
    disp(['Gradient descent. Step ',num2str(step-1),'. f = ',num2str(f_next),'.'])
    if step > maxSteps
        disp('Gradient descent. Maximum iterations reached.')
    elseif diff <= tol
        disp('Gradient descent. Cost function update less than tolerance.')
    end
end
xtraj(:,step:end) = repmat(NaN*ones(size(x,1),1),[1 size(xtraj(:,step:end),2)]);
disp(['Gauss-Newton. Stopped after ',num2str(step),' iterations.'])