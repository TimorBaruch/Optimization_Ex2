import numpy as np


def qp_function(x):
    f = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2
    grad = np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])
    hess = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    return f, grad, hess


# QP inequality constraints
def qp_ineq_constraints():
    def ineq1(x):
        return -x[0], np.array([-1, 0, 0]), np.zeros((3, 3))  # x >= 0

    def ineq2(x):
        return -x[1], np.array([0, -1, 0]), np.zeros((3, 3))  # y >= 0

    def ineq3(x):
        return -x[2], np.array([0, 0, -1]), np.zeros((3, 3))  # z >= 0

    return [ineq1, ineq2, ineq3]


# QP equality constraint
def qp_eq_constraint():
    return np.array([1, 1, 1]).reshape(1, 3)  # x + y + z = 1


def lp_function(x):
    f = -(x[0] + x[1])  # Maximize x + y is the same as minimizing -(x + y)
    grad = np.array([-1, -1])
    hess = np.zeros((2, 2))
    return f, grad, hess


# LP inequality constraints
def lp_ineq_constraints():
    def ineq1(x):
        return -x[1] - x[0] + 1, np.array([-1, -1]), np.zeros((2, 2))  # y >= -x + 1

    def ineq2(x):
        return -1 + x[1], np.array([0, 1]), np.zeros((2, 2))  # y <= 1

    def ineq3(x):
        return -2 + x[0], np.array([1, 0]), np.zeros((2, 2))  # x <= 2

    def ineq4(x):
        return -x[1], np.array([0, -1]), np.zeros((2, 2))  # y >= 0

    return [ineq1, ineq2, ineq3, ineq4]
