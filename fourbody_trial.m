% fourbody_trial.m
% Script to find multiple concave central configurations for the four-body problem
% using Newton's Method with grid sampling.

% Clear workspace and command window for a clean environment
clear;
clc;

% Define bounds for the variables [x3; x4; y3; y4]
lb = [0; -1; 1.73; 0];     % Lower bounds
ub = [1; 0; 3.73; 1.73];   % Upper bounds

% Define the number of divisions for each variable in the grid
num_divisions = [10, 10, 10, 10]; % Adjust for finer grid

% Generate grid points for each variable
x3_vals = linspace(lb(1), ub(1), num_divisions(1));
x4_vals = linspace(lb(2), ub(2), num_divisions(2));
y3_vals = linspace(lb(3), ub(3), num_divisions(3));
y4_vals = linspace(lb(4), ub(4), num_divisions(4));

% Create all combinations of initial guesses
[X3_grid, X4_grid, Y3_grid, Y4_grid] = ndgrid(x3_vals, x4_vals, y3_vals, y4_vals);
initial_guesses = [X3_grid(:), X4_grid(:), Y3_grid(:), Y4_grid(:)];

% Parameters for Newton's Method
max_iter = 100;     % Maximum number of iterations
tol = 1e-8;         % Tolerance for convergence

% Preallocate storage for solutions
num_guesses = size(initial_guesses, 1);
solutions = NaN(4, num_guesses);     % To store [x3; x4; y3; y4]
f34_values = NaN(1, num_guesses);   % To store f34 values
converged_flags = false(1, num_guesses); % To track convergence

% Open parallel pool for faster computation (optional)
% Uncomment the next line if you have the Parallel Computing Toolbox
% parpool;

% Loop over all initial guesses using parallel processing for efficiency
parfor i = 1:num_guesses
    x0 = initial_guesses(i, :)';
    [x_sol, converged] = newton_method(x0, max_iter, tol);
    if converged
        % Compute f34 to check acceptability
        f34 = compute_f34(x_sol);
        if abs(f34) < tol
            % Store the solution and f34 value
            solutions(:, i) = x_sol;
            f34_values(i) = f34;
            converged_flags(i) = true;
        end
    end
end

% Close parallel pool if it was opened
% Uncomment the next line if you opened the parallel pool
% delete(gcp);

% Extract only the converged and acceptable solutions
valid_indices = converged_flags;
solutions = solutions(:, valid_indices);
f34_values = f34_values(valid_indices);

% Remove duplicate solutions by rounding and using 'unique'
[unique_solutions, ia, ~] = unique(round(solutions', 8), 'rows');
unique_f34_values = f34_values(ia);

% Display the acceptable solutions
disp('Acceptable Solutions where abs(f34) < tolerance:');
for i = 1:size(unique_solutions, 1)
    x = unique_solutions(i, :)';
    disp(['Solution ', num2str(i), ':']);
    disp(['x3 = ', num2str(x(1))]);
    disp(['x4 = ', num2str(x(2))]);
    disp(['y3 = ', num2str(x(3))]);
    disp(['y4 = ', num2str(x(4))]);
    disp(['f34 = ', num2str(unique_f34_values(i))]);
    disp('---------------------------');
end

% -------------------------------------------------------------------------
% Local Function Definitions
% -------------------------------------------------------------------------

function [x, converged] = newton_method(x0, max_iter, tol)
    % newton_method performs Newton-Raphson iterations to find a root
    % Inputs:
    %   x0 - Initial guess (4x1 vector)
    %   max_iter - Maximum number of iterations
    %   tol - Tolerance for convergence
    % Outputs:
    %   x - Solution vector
    %   converged - Boolean indicating if convergence was achieved

    x = x0;
    converged = false;
    for iter = 1:max_iter
        F = myfun(x);            % Compute the residuals
        J = jacobian_num(x);     % Compute the Jacobian matrix
        % Check if Jacobian is singular
        if rcond(J) < eps
            break; % Singular Jacobian, cannot proceed
        end
        delta = -J \ F;           % Compute the update
        x = x + delta;            % Update the solution
        if norm(delta) < tol
            converged = true;     % Convergence achieved
            break;
        end
    end
end

function J = jacobian_num(x)
    % jacobian_num computes the Jacobian matrix numerically using finite differences
    % Input:
    %   x - Current solution vector (4x1)
    % Output:
    %   J - Jacobian matrix (4x4)

    epsilon = 1e-6;
    n = length(x);
    J = zeros(n);
    F0 = myfun(x);
    for i = 1:n
        x_eps = x;
        x_eps(i) = x_eps(i) + epsilon;
        F_eps = myfun(x_eps);
        J(:, i) = (F_eps - F0) / epsilon;
    end
end

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
    % compute_f34 computes f34 for a given solution
    % Input:
    %   x - vector of variables [x3; x4; y3; y4]
    % Output:
    %   f34 - value of f34

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
