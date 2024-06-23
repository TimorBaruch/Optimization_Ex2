import numpy as np


def interior_pt(f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    """Solves the constrained optimization problem using the interior point method with log-barrier"""
    path_hist = dict(path=[], values=[])
    t = 1

    def apply_log_barrier(x, t):
        """Combines the objective function and log-barrier term"""
        f_val, f_grad, f_hess = f(x)
        f_val, f_grad, f_hess = f(x)
        b_val, b_grad, b_hess = log_barrier(ineq_constraints, x)

        # if constraint value is non-negative, the barrier term return infinity, indicating infeasibility
        if b_val == np.inf:
            return np.inf, np.zeros_like(f_grad), np.zeros_like(f_hess)

        return t * f_val + b_val, t * f_grad + b_grad, t * f_hess + b_hess

    prev_f, prev_g, prev_h = apply_log_barrier(x0, t)
    prev_x0 = x0

    # save values for visualizing
    path_hist['path'].append(prev_x0.copy())
    path_hist['values'].append(f(prev_x0.copy())[0])

    while (len(ineq_constraints) / t) > 1e-8:
        for i in range(10):
            direction = find_direction(prev_h, eq_constraints_mat, prev_g)
            step_len = wolfe_condition_backtracking(f, prev_x0, prev_f, prev_g, direction, ineq_constraints, t)
            curr_x0 = prev_x0 + step_len * direction
            curr_f, curr_g, curr_h = apply_log_barrier(curr_x0, t)

            # handles cases where the constraint value is non-negative, reducing step size to find feasible point
            if np.isinf(curr_f):
                step_len *= 0.5
                continue

            lamda = np.sqrt(np.dot(direction, np.dot(curr_h, direction.T)))
            if 0.5 * (lamda ** 2) < 1e-8:
                # more iterations are unlikely to significantly improve the objective function value or achieve better convergence
                break

            # update values for next iteration
            prev_x0 = curr_x0
            prev_f = curr_f
            prev_g = curr_g
            prev_h = curr_h

        # saving values and paths for visualization
        path_hist['path'].append(prev_x0.copy())
        path_hist['values'].append(f(prev_x0.copy())[0])
        t *= 10

    return path_hist, prev_x0, f(prev_x0.copy())[0]


def wolfe_condition_backtracking(f, x, f_val, gradient, direction, ineq_constraints, t, alpha=0.01, beta=0.5,
                                 max_iter=10):
    """Performs backtracking line search to satisfy the Wolfe conditions"""
    step_len = 1
    x_new = x + step_len * direction
    val_curr, _, _ = f(x_new)

    iter_count = 0
    while iter_count < max_iter and val_curr > f_val + alpha * step_len * gradient.dot(direction):
        step_len *= beta
        x_new = x + step_len * direction
        val_curr, _, _ = f(x_new)
        iter_count += 1

    return step_len


def find_direction(prev_hess, eq_constraints_mat, prev_grad):
    """Computes the Newton direction considering equality constraints"""
    if eq_constraints_mat is not None and eq_constraints_mat.size > 0:

        # ensure eq_constraints_mat is a row vector if needed
        if len(eq_constraints_mat.shape) == 1:
            eq_constraints_mat = eq_constraints_mat[np.newaxis, :]

        # construct the block matrix
        top_left = prev_hess
        top_right = eq_constraints_mat.T
        bottom_left = eq_constraints_mat
        bottom_right = np.zeros((eq_constraints_mat.shape[0], eq_constraints_mat.shape[0]))

        left_mat = np.block([[top_left, top_right], [bottom_left, bottom_right]])
        right_vec = np.concatenate([-prev_grad, np.zeros(eq_constraints_mat.shape[0])])

        # solve the linear system left_mat ⋅ ans = right_vec
        ans = np.linalg.solve(left_mat, right_vec)

        # extract the direction vector (first n elements)
        direction = ans[:prev_hess.shape[1]]

    else:
        # unconstrained case, direction is calculated by solving the linear system prev_hess ⋅ ans = -prev_grad
        direction = np.linalg.solve(prev_hess, -prev_grad)

    return direction


def log_barrier(ineq_constraints, x):
    """Computes the log-barrier term for inequality constraints"""
    barrier_val = 0
    barrier_grad = np.zeros_like(x)
    barrier_hess = np.zeros((len(x), len(x)))

    for constraint in ineq_constraints:
        c_val, c_grad, c_hess = constraint(x)

        # handles cases where the constraint value is non-negative (violated) by returning infinity
        if c_val >= 0:
            return np.inf, np.zeros_like(x), np.zeros((len(x), len(x)))

        barrier_val += np.log(-c_val)
        barrier_grad += c_grad / (-c_val)

        grad = c_grad / c_val
        barrier_hess += (c_hess * c_val - np.outer(grad, grad)) / (c_val ** 2)

    return -barrier_val, barrier_grad, -barrier_hess
