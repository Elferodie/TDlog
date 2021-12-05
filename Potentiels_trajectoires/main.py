from create_potentials import *
from generate_trajectories import *
from plot_functions import *


def main():
    # Creation du potentiel
    beta = 3
    potential = create_spiral_potential(beta, inter_space=3, r_min=0.6, r_max=5, height=-1, variance=0.4)
    # Visualisation du potentiel
    grid, potential_on_grid = potential.create_potential_on_grid(precision=100)
    plot_potential(grid, potential_on_grid, figsize=(10, 4), save_name=f"Potentiel en spiral", show=True)

    # Parametres de trajectoire
    delta_t = 0.01
    nb_points = 9999
    x_0 = np.array([0, 0])
    # Generation de la trajectoire
    trajectory, _ = UnbiasedTraj(potential, x_0, delta_t=delta_t, nb_points=nb_points, save=1, save_energy=False, seed=0)
    # Visualisation de la trajectoire
    plot_trajectory(grid, potential_on_grid, trajectory, figsize=(13, 4), save_name=f"Trajectoire pour un potentiel en spiral béta={beta}", show=True)


if __name__ == "__main__":
    main()