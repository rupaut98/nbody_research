% Clear workspace and command window for a clean environment
clear;
clc;

% Define bounds for the variables
lb = [0; -1; 1.73; 0];     % Lower bounds [x3_min; x4_min; y3_min; y4_min]
ub = [1; 0; 3.73; 1.73];   % Upper bounds [x3_max; x4_max; y3_max; y4_max]

% Set options for lsqnonlin
options = optimoptions('lsqnonlin', ...
                       'Display', 'off', ...
                       'TolFun', 1e-12, ...
                       'TolX', 1e-12, ...
                       'MaxIterations', 1000, ...
                       'MaxFunctionEvaluations', 2000);

% Set up problem structure for lsqnonlin
problem = createOptimProblem('lsqnonlin', ...
    'objective', @myfun, ...
    'x0', [], ...
    'lb', lb, ...
    'ub', ub, ...
    'options', options);

% Use MultiStart to find multiple solutions
ms = MultiStart('Display', 'off', 'UseParallel', false);

% Number of starting points
num_starting_points = 10000;

% Generate random starting points within bounds
rng('default'); % For reproducibility
start_points = lb + rand(length(lb), num_starting_points) .* (ub - lb);

% Preallocate for solutions
solutions = [];
fun_values = [];
f34_values = [];

% Run MultiStart
for i = 1:num_starting_points
    problem.x0 = start_points(:, i);
    [x, resnorm, residual, exitflag, output] = run(ms, problem, 1);
    solutions = [solutions, x];
    fun_values = [fun_values, resnorm];
    % Compute f34 for the solution
    f34 = compute_f34(x);
    f34_values = [f34_values, f34];
end

% Remove duplicate solutions
[unique_solutions, ia, ~] = unique(round(solutions', 8), 'rows');
unique_fun_values = fun_values(ia);
unique_f34_values = f34_values(ia);

% Set tolerance for f34
tolerance = 1e-8;

% Filter solutions where abs(f34) is less than tolerance
acceptable_indices = abs(unique_f34_values) < tolerance;

% Display all unique solutions
disp('All Unique Solutions:');
for i = 1:size(unique_solutions, 1)
    x = unique_solutions(i, :)';
    disp(['Solution ', num2str(i), ':']);
    disp(['x3 = ', num2str(x(1))]);
    disp(['x4 = ', num2str(x(2))]);
    disp(['y3 = ', num2str(x(3))]);
    disp(['y4 = ', num2str(x(4))]);
    disp(['Residual Sum of Squares: ', num2str(unique_fun_values(i))]);
    disp(['f34 = ', num2str(unique_f34_values(i))]);
    disp('---------------------------');
end

% Display the acceptable solutions
disp('Acceptable Solutions where abs(f34) < tolerance:');
for i = find(acceptable_indices)'
    x = unique_solutions(i, :)';
    disp(['Solution ', num2str(i), ':']);
    disp(['x3 = ', num2str(x(1))]);
    disp(['x4 = ', num2str(x(2))]);
    disp(['y3 = ', num2str(x(3))]);
    disp(['y4 = ', num2str(x(4))]);
    disp(['Residual Sum of Squares: ', num2str(unique_fun_values(i))]);
    disp(['f34 = ', num2str(unique_f34_values(i))]);
    disp('---------------------------');
end

% -------------------------------------------------------------------------
% Local Function Definitions
% -------------------------------------------------------------------------
function F = myfun(x)
    % myfun computes the residuals of the system of equations
    % Input:
    %   x - vector of variables [x3; x4; y3; y4]
    % Output:
    %   F - vector of residuals [f12; f13; f24; f34]

    x3 = x(1);
    x4 = x(2);
    y3 = x(3);
    y4 = x(4);

    % Compute common terms to simplify expressions
    term_a = ((-1 - x3)^2 + y3^2)^(-3/2);
    term_b = ((1 - x3)^2 + y3^2)^(-3/2);
    term_c = ((-1 - x4)^2 + y4^2)^(-3/2);
    term_d = ((1 - x4)^2 + y4^2)^(-3/2);
    term_e = ((x3 - x4)^2 + (y3 - y4)^2)^(-3/2);
    term_f = ((x3 - 1)^2 + y3^2)^(-3/2);

    % f12 equation
    f12 = 2 * (term_a - term_b) * y3 + 2 * (5 * term_c - 5 * term_d) * y4;

    % f13 equation
    f13 = -2 * (0.25 - 2 * term_b) * y3 + (5 * term_c - 5 * term_e) * ((x4 + 1)*(y4 - y3) + y3*(x3 - x4));

    % f24 equation
    f24 = 2 * (0.375 - 3 * term_c) * y4 + (term_f - term_e) * (-y3*(1 - x4) - y4*(x3 - 1));

    % f34 equation
    f34 = (3 * term_a - 3 * term_c) * ((x4 + 1)*(y4 - y3) + y3*(x3 - x4)) + ...
          (2 * term_b - 2 * term_d) * (y3*(1 - x4) + y4*(x3 - 1));

    % Return the vector of residuals
    F = [f12; f13; f24; f34];
end

function f34 = compute_f34(x)
    % compute_f34 computes f34 for given x

    x3 = x(1);
    x4 = x(2);
    y3 = x(3);
    y4 = x(4);

    % Compute common terms to simplify expressions
    term_a = ((-1 - x3)^2 + y3^2)^(-3/2);
    term_b = ((1 - x3)^2 + y3^2)^(-3/2);
    term_c = ((-1 - x4)^2 + y4^2)^(-3/2);
    term_d = ((1 - x4)^2 + y4^2)^(-3/2);
    term_e = ((x3 - x4)^2 + (y3 - y4)^2)^(-3/2);
    term_f = ((x3 - 1)^2 + y3^2)^(-3/2);

    % f34 equation
    f34 = (3 * term_a - 3 * term_c) * ((x4 + 1)*(y4 - y3) + y3*(x3 - x4)) + ...
          (2 * term_b - 2 * term_d) * (y3*(1 - x4) + y4*(x3 - 1));
end
