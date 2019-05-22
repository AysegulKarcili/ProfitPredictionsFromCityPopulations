clear ; close all; 
% THIS PROGRAM MODELS THE PROFIT OBTAINED AT DIFFERENT CITIES WITH RESPECT
% TO THEIR POPULATION SIZE. THIS ProgramMainFile CALLS THE ComputeCost()
% AND GradientDescent() FUNCTIONS.. 

% Load data
data = load('data.txt');
X = data(:, 1); % population size in 10,000s
y = data(:, 2); % profit in $10,000s

% Visualize data
figure;
plot(X, y, 'r*'); 
xlabel('population'); ylabel('revenue');

% Preparation for Linear Regression
m = length(y); % number of training examples
X = [ones(m, 1), X]; % Add a column of ones to X
theta = zeros(2, 1); % Initialize parameters

% Some gradient descent settings
num_iterations = 1500;
alpha = 0.01;

% Compute the initial cost
J = ComputeCost(X, y, theta);
fprintf('With theta = [0 ; 0] the computed cost is %f\n', J);

% Train theta parameters by running gradient descent
theta = GradientDescent(X, y, theta, alpha, num_iterations);
fprintf('Theta trained by gradient descent: [ %f  %f ]'' \n', theta(1), theta(2));

% Plot the linear fit together with data points
figure; 
plot(X(:,2), y, 'r*'); 
xlabel('population'); ylabel('revenue');
hold on; 
plot(X(:,2), X*theta, '-');
legend('Training data', 'Linear regression', 'Location','southeast');
hold off;

% For demo, predict profits for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] * theta;
fprintf('For population = 35,000, profit prediction is %f\n', predict1 * 10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, profit prediction is %f\n', predict2 * 10000);

% Visualize cost over a range of theta values i.e. J(theta) or J(theta0,theta1):

% Grid over which J will be calculated
theta0vals = linspace(-10, 10, 100);
theta1vals = linspace(-1, 4, 100);

% Calculate costs over a range of theta 
J_values = zeros(length(theta0vals), length(theta1vals));
for i = 1:length(theta0vals)
    for j = 1:length(theta1vals)
	  thetaValue = [theta0vals(i); theta1vals(j)];
	  J_values(i,j) = ComputeCost(X, y, thetaValue);
    end
end

% Surface plot of cost values
figure;
surf(theta0vals, theta1vals, J_values'); % transpose due to the structure of surf() 
xlabel('\theta_0'); ylabel('\theta_1'); zlabel('Cost Value')

% Contour plot of cost values
figure;
contour(theta0vals, theta1vals, J_values', logspace(-2, 3, 20)) % Plot contours spaced logarithmically between 0.01 and 1000
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
