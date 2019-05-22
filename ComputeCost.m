function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = computeCost(X, y, theta) computes the cost of linear fit of the
%   given theta parameters w.r.t. actual y values.

m = length(y); % number of training examples

J = (1/(2*m))*sum((X*theta - y).^2); % Cost is half of the average squared distance between the predicted value and actual value.

end
