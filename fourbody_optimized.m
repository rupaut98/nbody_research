% Comment for Nishan: lower bound and upper bound can be changed to fit the problem
%it can be changed from the lines 16-17
%if you would like to change the masses, it would need to be changed from two functions
% function F_original = compute_residuals(x) in lines (246-249)
% function function f34 = compute_f34(x) in lines (294-297)

% fourbody_trial_optimized.m
% Script to find multiple concave central configurations for the four-body problem
% using Newton's Method with grid sampling and constraints.

% Clear workspace and command window for a clean environment
clear;
clc;

% Define expanded bounds for the variables [x3; x4; y3; y4]
lb = [0; -1; 1.73; 0];     % Lower bounds with non-zero minima
ub = [1; 0; 3.73; 1.73];           % Upper bounds expanded to include previous solutions

% Define the number of divisions for each variable in the grid
num_divisions = [20, 20, 20, 20]; % Increased from 10 to 20 divisions per variable

% Generate grid points for each variable
x3_vals = linspace(lb(1), ub(1), num_divisions(1)); %--> linspace(start, end, number of divisions)
x4_vals = linspace(lb(2), ub(2), num_divisions(2)); % so for x3 there would be 20 divisions between 0 and 1
y3_vals = linspace(lb(3), ub(3), num_divisions(3));
y4_vals = linspace(lb(4), ub(4), num_divisions(4));


% Create all combinations of initial guesses
[X3_grid, X4_grid, Y3_grid, Y4_grid] = ndgrid(x3_vals, x4_vals, y3_vals, y4_vals); %This will create a grid of values 
grid_guesses = [X3_grid(:), X4_grid(:), Y3_grid(:), Y4_grid(:)];

% Generate additional random initial guesses
num_random_guesses = 10000; % Adjust based on computational resources
random_guesses = lb' + rand(num_random_guesses, 4) .* (ub' - lb');

% Combine grid and random guesses
initial_guesses = [grid_guesses; random_guesses];

% Parameters for Newton's Method
max_iter = 100;     % Maximum number of iterations
tol = 1e-5;         % Tolerance for convergence

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
    [x_sol, converged] = newton_method(x0, max_iter, tol, lb, ub);
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

fprintf('Number of converged solutions: %d\n', size(solutions, 2));

% Remove duplicate solutions with enhanced filtering
tolerance = 1e-4; % Define a tolerance for considering solutions as duplicates

% Initialize empty arrays for unique solutions and their f34 values
unique_solutions = [];
unique_f34_values = [];

for i = 1:size(solutions, 2)
    sol = solutions(:, i);
    
    if isempty(unique_solutions)
        unique_solutions = sol';
        unique_f34_values = f34_values(i);
    else
        % Calculate Euclidean distance to existing unique solutions
        distances = sqrt(sum((unique_solutions - sol').^2, 2));
        
        % If the solution is farther than the tolerance from all unique solutions, add it
        if all(distances > tolerance)
            unique_solutions = [unique_solutions; sol'];
            unique_f34_values = [unique_f34_values; f34_values(i)];
        end
    end
end

fprintf('Number of unique solutions after filtering: %d\n', size(unique_solutions, 1));

% Define additional filtering criteria
min_x3 = 1e-3; % Minimum allowable x3
min_y4 = 1e-3; % Minimum allowable y4

% Apply filters
filtered_indices = (unique_solutions(:, 1) >= min_x3) & (unique_solutions(:, 4) >= min_y4);
filtered_solutions = unique_solutions(filtered_indices, :);
filtered_f34_values = unique_f34_values(filtered_indices);

fprintf('Number of filtered solutions: %d\n', size(filtered_solutions, 1));

% Display the filtered acceptable solutions
disp('Filtered Acceptable Solutions where abs(f34) < tolerance:');
for i = 1:size(filtered_solutions, 1)
    x = filtered_solutions(i, :)';
    disp(['Solution ', num2str(i), ':']);
    disp(['x3 = ', num2str(x(1))]);
    disp(['x4 = ', num2str(x(2))]);
    disp(['y3 = ', num2str(x(3))]);
    disp(['y4 = ', num2str(x(4))]);
    disp(['f34 = ', num2str(filtered_f34_values(i))]);
    disp('---------------------------');
end

% -------------------------------------------------------------------------
% Local Function Definitions
% -------------------------------------------------------------------------

function [x, converged] = newton_method(x0, max_iter, tol, lb, ub)
    % newton_method performs Newton-Raphson iterations with line search and bounds checking
    % Inputs:
    %   x0 - Initial guess (4x1 vector)
    %   max_iter - Maximum number of iterations
    %   tol - Tolerance for convergence
    %   lb - Lower bounds (4x1 vector)
    %   ub - Upper bounds (4x1 vector)
    % Outputs:
    %   x - Solution vector
    %   converged - Boolean indicating if convergence was achieved

    x = x0;
    converged = false;

    for iter = 1:max_iter
        F = myfun(x);            % Compute residuals
        J = jacobian_num(x);     % Compute Jacobian matrix

        % Check if Jacobian is singular
        if rcond(J) < eps
            break; % Singular Jacobian, cannot proceed
        end

        delta = -J \ F;           % Compute update

        alpha = 1; % Reset alpha at each iteration

        % Line search to ensure sufficient decrease and within bounds
        while true
            x_new = x + alpha * delta;

            % Check if x_new is within bounds
            if any(x_new < lb) || any(x_new > ub)
                % Step goes out of bounds, reduce alpha
                alpha = alpha / 2;
                if alpha < 1e-4
                    break; % Give up on this step
                end
                continue; % Try with reduced alpha
            end

            F_new = myfun(x_new);

            if norm(F_new) < norm(F) || alpha < 1e-4
                break; % Accept the step
            end

            alpha = alpha / 2; % Reduce step size and retry
        end

        x = x_new; % Update solution

        if norm(delta) < tol
            converged = true;
            break;
        end
    end
end

function J = jacobian_num(x)
    % jacobian_num computes the Jacobian matrix numerically using central differences
    % Input:
    %   x - Current solution vector (4x1)
    % Output:
    %   J - Jacobian matrix (4x4)

    epsilon = 1e-6;
    n = length(x);          % Number of variables (4)
    F0 = myfun(x);          % Current residuals (4x1)
    m = length(F0);         % Number of residuals (4)
    J = zeros(m, n);        % Initialize Jacobian (4x4)

    for i = 1:n
        x_eps_plus = x;
        x_eps_minus = x;
        x_eps_plus(i) = x_eps_plus(i) + epsilon;
        x_eps_minus(i) = x_eps_minus(i) - epsilon;
        F_plus = myfun(x_eps_plus);
        F_minus = myfun(x_eps_minus);
        J(:, i) = (F_plus - F_minus) / (2 * epsilon); % Central difference
    end
end

function F = myfun(x)
    % myfun computes the residuals of the system of equations
    % Input:
    %   x - vector of variables [x3; x4; y3; y4]
    % Output:
    %   F - vector of residuals [f12; f13; f24; f34]

    F_original = compute_residuals(x);
    
    % Return only the original residuals without penalties
    F = F_original;
end

function F_original = compute_residuals(x)
    % compute_residuals computes the original residuals without penalties
    % Input:
    %   x - vector of variables [x3; x4; y3; y4]
    % Output:
    %   F_original - vector of residuals [f12; f13; f24; f34]
    
    % Unpack variables
    x3 = x(1);
    x4 = x(2);
    y3 = x(3);
    y4 = x(4);
    
    % Define masses
    m1 = 5;
    m2 = 4;
    m3 = 3;
    m4 = 5;
    
    % Precompute distances and their inverses raised to the power -3/2
    term_a = ((-1 - x3)^2 + y3^2)^(-3/2);
    term_b = ((1 - x3)^2 + y3^2)^(-3/2);
    term_c = ((-1 - x4)^2 + y4^2)^(-3/2);
    term_d = ((1 - x4)^2 + y4^2)^(-3/2);
    term_e = ((x3 - x4)^2 + (y3 - y4)^2)^(-3/2);
    term_f = ((x3 - 1)^2 + y3^2)^(-3/2);
    
    % For f12:
    f12 = 2 * m3 * (term_a - term_b) * y3 + 2 * m4 * (term_c - term_d) * y4;
    
    % For f13:
    term_constant = 1/8; % sqrt(4)/16 simplifies to 1/8
    f13 = -2 * m2 * (term_constant - term_b) * y3 + ...
          m4 * (term_c - term_e) * ((x4 + 1) * (y4 - y3) + y3 * (x3 - x4));
    
    % For f24:
    f24 = 2 * m1 * (term_constant - term_c) * y4 + ...
          m3 * (term_f - term_e) * (-y3 * (1 - x4) - y4 * (x3 - 1));
    
    % For f34:
    f34 = m1 * (term_a - term_c) * ((x4 + 1) * (y4 - y3) + y3 * (x3 - x4)) + ...
          m2 * (term_b - term_d) * (y3 * (1 - x4) + y4 * (x3 - 1));
    
    % Combine residuals
    F_original = [f12; f13; f24; f34];
end


function f34 = compute_f34(x)
    % compute_f34 computes f34 for a given solution
    % Input:
    %   x - vector of variables [x3; x4; y3; y4]
    % Output:
    %   f34 - value of f34

    % Unpack variables
    x3 = x(1);
    x4 = x(2);
    y3 = x(3);
    y4 = x(4);

    % Define masses
    m1 = 5;
    m2 = 4;
    m3 = 3;
    m4 = 5;

    % Precompute distances and their inverses raised to the power -3/2
    term_a = ((-1 - x3)^2 + y3^2)^(-3/2);
    term_b = ((1 - x3)^2 + y3^2)^(-3/2);
    term_c = ((-1 - x4)^2 + y4^2)^(-3/2);
    term_d = ((1 - x4)^2 + y4^2)^(-3/2);
    term_e = ((x3 - x4)^2 + (y3 - y4)^2)^(-3/2);
    term_f = ((x3 - 1)^2 + y3^2)^(-3/2);

    % Compute f34 using the updated equation with masses
    f34 = m1 * (term_a - term_c) * ((x4 + 1) * (y4 - y3) + y3 * (x3 - x4)) + ...
          m2 * (term_b - term_d) * (y3 * (1 - x4) + y4 * (x3 - 1));
end

