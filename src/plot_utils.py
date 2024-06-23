import matplotlib.pyplot as plt
import numpy as np


def plot_objective_values(values, problem_type):
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(values))

    ax.plot(x, values, marker='o', linestyle='-', color='b', markersize=7, linewidth=2)
    ax.set_title(f'Objective Value vs. Outer Iteration Number for {problem_type}', fontsize=14)
    ax.set_xlabel('#Iteration', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_path_qp(path, plane_color='gray', path_color='brown', final_color='purple'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)

    # plotting the reference plane
    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color=plane_color, alpha=0.3)

    # plotting the path
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path', color=path_color, linewidth=2)

    # plotting the final candidate point
    ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=150, c=final_color, marker='o', label='The candidate')

    ax.set_title("Feasible Region and Path by the Algorithm QP", fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.view_init(45, 45)

    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_path_lp(path, plane_color='gray', path_color='black', final_color='purple'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    path = np.array(path)

    x = np.linspace(-1, 3, 1000)
    y = np.linspace(-2, 2, 1000)

    # dictionary of inequality constraints for LP
    constraints_ineq = {
        'y=0': (x, x * 0),
        'y=1': (x, x * 0 + 1),
        'x=2': (y * 0 + 2, y),
        'y=-x+1': (x, -x + 1)
    }

    # plotting inequality constraints for LP
    for label, (x_data, y_data) in constraints_ineq.items():
        ax.plot(x_data, y_data, label=label, linewidth=2)

    # plotting the feasible region for LP
    ax.fill([0, 2, 2, 1], [1, 1, 0, 0], color=plane_color, alpha=0.3, label='Feasible region')

    # plotting the path taken in LP
    ax.plot(path[:, 0], path[:, 1], color=path_color, label='Path', linewidth=2)

    # plotting the final candidate point for LP
    ax.scatter(path[-1][0], path[-1][1], s=150, c=final_color, marker='o', label='The candidate')

    ax.set_title("Feasible region and path by the LP algorithm", fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.legend(fontsize=12, loc='lower left')
    plt.tight_layout()
    plt.show()