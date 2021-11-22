import numpy as np


def UnbiasedTraj(potential, X0, delta_t=1e-3, nb_points=1000, save=1, save_energy=False, seed=0):
    """Simulates an overdamped langevin trajectory with Euler-Maruyama integration scheme

    :param potential: potential object, must have methods for energy gradientand energy evaluation
    :param X0: Initial position, must be a 2D vector
    :param delta_t: time step size
    :param nb_points: Number of points in the trajectory
    :param save: int param correspond to the frequency (in number of step) at which the trajectory is saved
    :param save_energy: Bool parameter to save energy along the trajectory

    :return: traj: np.array with ndim = 2 and shape = (T // save + 1, 2)
    :return: potential_values: np.array with ndim = 2 and shape = (T // save + 1, 1)
    """
    r = np.random.RandomState(seed)
    X = X0
    dim = X.shape[0]
    traj = [X]
    if save_energy:
        potential_values = [potential.V(X)]
    else:
        potential_values = None

    for i in range(nb_points):
        binom = r.normal(size=(dim,))
        X = X - potential.nabla_V(X) * delta_t + np.sqrt(2 * delta_t / potential.beta) * binom
        if i % save == 0:
            traj.append(X)
            if save_energy:
                potential_values.append(potential.V(X))

    return np.array(traj), np.array(potential_values)
