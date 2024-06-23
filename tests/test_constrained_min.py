import unittest
import numpy as np
from src.constrained_min import interior_pt
from tests.examples import qp_function, qp_ineq_constraints, qp_eq_constraint, lp_function, lp_ineq_constraints
from src.plot_utils import plot_path_qp, plot_path_lp, plot_objective_values


def test_optimization(problem_type, objective_function, inequality_constraints, equality_constraint=None,
                      initial_point=None):
    hist_dict, candidate, obj_val = interior_pt(objective_function, inequality_constraints, equality_constraint, 0,
                                                initial_point)

    if problem_type not in ['QP', 'LP']:
        raise ValueError("Unsupported problem type. Supported types are 'QP' and 'LP'.")

    # Evaluate and print results
    coordinates = ', '.join(f'{dim}={val}' for dim, val in zip('xyz', candidate))
    ineq_constraints_at_final = [c(candidate)[0] for c in inequality_constraints]

    print(f"Final candidate: {coordinates}")
    print(f"Objective value at the final candidate: {obj_val}")
    print(f"Inequality constraints values at the final candidate: {ineq_constraints_at_final}")

    if equality_constraint is not None and candidate is not None:
        equality_values = np.dot(equality_constraint, candidate)[0]
        print(f"Equality constraints values at the final candidate: {equality_values}")

    # Plot objective values and optimization path
    if problem_type == 'LP':
        plot_path_lp(hist_dict['path'])
    else:
        plot_path_qp(hist_dict['path'])
    plot_objective_values(hist_dict['values'], problem_type)


class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        test_optimization('QP', qp_function, qp_ineq_constraints(), qp_eq_constraint(), np.array([0.1, 0.2, 0.7]))

    def test_lp(self):
        test_optimization('LP', lp_function, lp_ineq_constraints(), initial_point=np.array([0.5, 0.75]))


if __name__ == '__main__':
    unittest.main()
