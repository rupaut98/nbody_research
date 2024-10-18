% Comment for Nishan: lower bound and upper bound can be changed to fit the problem
% It can be changed from the lines 16-17
% If you would like to change the masses, it would need to be changed in two functions:
% - function F_original = compute_residuals(x)
% - function f14 = compute_f14(x)

% fourbody_trial_optimized.m
% Script to find multiple concave central configurations for the four-body problem
% using fsolve with grid sampling and constraints.

% Clear workspace and command window for a clean environment
clear;
clc;

% Define expanded bounds for the variables [x3; x4; y3; y4]
lb = [1e-3; -1; 1.73; 1e-3];     % Lower bounds with non-zero minima
ub = [3; 1; 5; 3.73];            % Upper bounds expanded to include previous solutions

% Define the number of divisions for each variable in the grid
num_divisions = [20, 20, 20, 20]; % Increased from 10 to 20 divisions per variable

% Generate grid points for each variable
x3_vals = linspace(lb(1), ub(1), num_divisions(1)); % linspace(start, end, number of divisions)
x4_vals = linspace(lb(2), ub(2), num_divisions(2));
y3_vals = linspace(lb(3), ub(3), num_divisions(3));
y4_vals = linspace(lb(4), ub(4), num_divisions(4));

% Create all combinations of initial guesses
[X3_grid, X4_grid, Y3_grid, Y4_grid] = ndgrid(x3_vals, x4_vals, y3_vals, y4_vals);
grid_guesses = [X3_grid(:), X4_grid(:), Y3_grid(:), Y4_grid(:)];

% Generate additional random initial guesses
num_random_guesses = 10000; % Adjust based on computational resources
random_guesses = lb' + rand(num_random_guesses, 4) .* (ub' - lb');

% Combine grid and random guesses
initial_guesses = [grid_guesses; random_guesses];

% Define number of guesses
num_guesses = size(initial_guesses, 1);

% Parameters for Newton's Method and fsolve
max_iter = 100;     % Maximum number of iterations
tol = 1e-5;         % Tolerance for convergence

% Parameters for fsolve
options = optimoptions('fsolve', 'Display', 'off', ...
    'MaxIterations', 1000, 'MaxFunctionEvaluations', 5000, 'FunctionTolerance', tol);

% Preallocate storage for solutions
solutions = cell(num_guesses, 1);
f14_values = zeros(num_guesses, 1);
converged_flags = false(num_guesses, 1);

% Loop over all initial guesses using parallel processing for efficiency
parfor i = 1:num_guesses
    x0 = initial_guesses(i, :)';

    % Run fsolve
    [x_sol, fval, exitflag] = fsolve(@myfun, x0, options);

    if exitflag > 0 && all(abs(fval) < tol)
        % Compute f14 to check acceptability
        f14 = compute_f14(x_sol);
        if abs(f14) < tol
            % Store the solution and f14 value
            solutions{i} = x_sol;
            f14_values(i) = f14;
            converged_flags(i) = true;
        end
    end
end

% Extract only the converged and acceptable solutions
valid_indices = converged_flags;
solutions = solutions(valid_indices);
f14_values = f14_values(valid_indices);

% Convert solutions from cell array to matrix
if ~isempty(solutions)
    solutions = cell2mat(solutions')';
else
    solutions = [];
end

fprintf('Number of converged solutions: %d\n', size(solutions, 2));

% Remove duplicate solutions with enhanced filtering
tolerance = 1e-4; % Define a tolerance for considering solutions as duplicates

% Initialize empty arrays for unique solutions and their f14 values
unique_solutions = [];
unique_f14_values = [];

for i = 1:size(solutions, 2)
    sol = solutions(:, i);
    
    if isempty(unique_solutions)
        unique_solutions = sol';
        unique_f14_values = f14_values(i);
    else
        % Calculate Euclidean distance to existing unique solutions
        distances = sqrt(sum((unique_solutions - sol').^2, 2));
        
        % If the solution is farther than the tolerance from all unique solutions, add it
        if all(distances > tolerance)
            unique_solutions = [unique_solutions; sol'];
            unique_f14_values = [unique_f14_values; f14_values(i)];
        end
    end
end

fprintf('Number of unique solutions after filtering: %d\n', size(unique_solutions, 1));

% Define additional filtering criteria
min_x3 = 1e-3; % Minimum allowable x3
min_y4 = 1e-3; % Minimum allowable y4

% Apply filters
if ~isempty(unique_solutions)
    % Apply filters
    filtered_indices = (unique_solutions(:, 1) >= min_x3) & (unique_solutions(:, 4) >= min_y4);
    filtered_solutions = unique_solutions(filtered_indices, :);
    filtered_f14_values = unique_f14_values(filtered_indices);
else
    % No solutions found
    filtered_solutions = [];
    filtered_f14_values = [];
    fprintf('No solutions found to apply filters.\n');
end

fprintf('Number of filtered solutions: %d\n', size(filtered_solutions, 1));

% Display the filtered acceptable solutions
disp('Filtered Acceptable Solutions where abs(f14) < tolerance:');
for i = 1:size(filtered_solutions, 1)
    x = filtered_solutions(i, :)';
    disp(['Solution ', num2str(i), ':']);
    disp(['x3 = ', num2str(x(1))]);
    disp(['x4 = ', num2str(x(2))]);
    disp(['y3 = ', num2str(x(3))]);
    disp(['y4 = ', num2str(x(4))]);
    disp(['f14 = ', num2str(filtered_f14_values(i))]);
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

    m1 = 2;
    m2 = 1;
    m3 = 1;
    m4 = 1;

    % Unpack variables
    x3 = x(1);
    x4 = x(2);
    y3 = x(3);
    y4 = x(4);
    
    epsilon = 1e-8; % Small value to prevent division by zero or negative roots

    % % Compute terms with exponents expressed as (-3/2)
    % term_a = ((-1 - x3)^2 + y3^2 + epsilon)^(-3/2);
    % term_b = ((1 - x3)^2 + y3^2 + epsilon)^(-3/2);
    % term_c = ((-1 - x4)^2 + y4^2 + epsilon)^(-3/2);
    % term_d = ((1 - x4)^2 + y4^2 + epsilon)^(-3/2);
    % term_e = ((x3 - x4)^2 + (y3 - y4)^2 + epsilon)^(-3/2);
    % term_f = ((x3 - 1)^2 + y3^2 + epsilon)^(-3/2);

    % % For f12:
    % f12 = 2 * m3 * (term_a - term_b) * y3 + ...
    %       2 * m4 * (term_c - term_d) * y4;

    % % For f13:
    % f13 = -2 * m2 * (1/8 - term_b) * y3 + ...
    %       m4 * (term_c - term_e) * ((x4 + 1)*(y4 - y3) + y3*(x3 - x4));

    % % For f24:
    % f24 = 2 * m1 * (1/8 - term_c) * y4 + ...
    %       m3 * (term_f - term_e) * (-y3*(1 - x4) - y4*(x3 - 1));

    % % For f34:
    % f34 = m1 * (term_a - term_c) * ((x4 + 1)*(y4 - y3) + y3*(x3 - x4)) + ...
    %       m2 * (term_b - term_d) * (y3*(1 - x4) + y4*(x3 - 1));

    f12 = 2 * m3 * (((-1 - x3) ^ 2 + y3 ^ 2) ^ (-0.3e1 / 0.2e1) - ((1 - x3) ^ 2 + y3 ^ 2) ^ (-0.3e1 / 0.2e1)) * y3 + 2 * m4 * (((-1 - x4) ^ 2 + y4 ^ 2) ^ (-0.3e1 / 0.2e1) - ((1 - x4) ^ 2 + y4 ^ 2) ^ (-0.3e1 / 0.2e1)) * y4;
    f13 = -0.2e1 * m2 * (sqrt(0.4e1) / 0.16e2 - (((1 - x3) ^ 2 + y3 ^ 2) ^ (-0.3e1 / 0.2e1))) * y3 + (m4 * (((-1 - x4) ^ 2 + y4 ^ 2) ^ (-0.3e1 / 0.2e1) - ((x3 - x4) ^ 2 + (y3 - y4) ^ 2) ^ (-0.3e1 / 0.2e1)) * ((x4 + 1) * (y4 - y3) + y4 * (x3 - x4)));
    f24 = 0.2e1 * m1 * (sqrt(0.4e1) / 0.16e2 - (((-1 - x4) ^ 2 + y4 ^ 2) ^ (-0.3e1 / 0.2e1))) * y4 + (m3 * (((x3 - 1) ^ 2 + y3 ^ 2) ^ (-0.3e1 / 0.2e1) - ((x3 - x4) ^ 2 + (y3 - y4) ^ 2) ^ (-0.3e1 / 0.2e1)) * (-y3 * (1 - x4) - y4 * (x3 - 1)));
    f34 = m1 * (((-1 - x3) ^ 2 + y3 ^ 2) ^ (-0.3e1 / 0.2e1) - ((-1 - x4) ^ 2 + y4 ^ 2) ^ (-0.3e1 / 0.2e1)) * ((x4 + 1) * (y4 - y3) + y4 * (x3 - x4)) + m2 * (((1 - x3) ^ 2 + y3 ^ 2) ^ (-0.3e1 / 0.2e1) - ((1 - x4) ^ 2 + y4 ^ 2) ^ (-0.3e1 / 0.2e1)) * (y3 * (1 - x4) + y4 * (x3 - 1));



    % Combine residuals
    F_original = [f12; f13; f24; f34];
end

function f14 = compute_f14(x)
    % compute_f14 computes f14 for a given solution
    % Input:
    %   x - vector of variables [x3; x4; y3; y4]
    % Output:
    %   f14 - value of f14

    m1 = 2;
    m2 = 1;
    m3 = 1;
    m4 = 1;

    % Unpack variables
    x3 = x(1);
    x4 = x(2);
    y3 = x(3);
    y4 = x(4);
    
    f14 = -0.2e1 * m2 * (sqrt(0.4e1) / 0.16e2 - (((1 - x4) ^ 2 + y4 ^ 2) ^ (-0.3e1 / 0.2e1))) * y4 + (m3 * (((-1 - x3) ^ 2 + y3 ^ 2) ^ (-0.3e1 / 0.2e1) - ((x3 - x4) ^ 2 + (y3 - y4) ^ 2) ^ (-0.3e1 / 0.2e1)) * (-(x4 + 1) * (y4 - y3) - y4 * (x3 - x4)));
end
