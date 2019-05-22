function theta = gradientDescent(X, y, theta, alpha, num_iterations)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = gradientDescent(X, y, theta, alpha, num_iterations) updates
%   theta by taking num_iterations gradient steps with learning rate alpha.

m = length(y); % number of training examples

for i = 1:num_iterations
    theta = theta - alpha * (1/m) * (X'*(X*theta-y)); % Calculates the steepest direction (gradient of the cost function) in every step and descents into that direction. 
end

end
