import numpy as np
from scipy import integrate


def g(X, X0, height, variance):
    """Gaussian function

    :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)
    :param X0: np.array, Position  vector (x,y), ndim = 1, shape = (2,)
    :return: float, distance(X, X0)
    """
    return height * np.exp(- ( (X[0]-X0[0])**2 + (X[1]-X0[1])**2 ) / (2 * variance))


class WellPotential:
    """Class to gather methods related to the potential function"""

    def __init__(self, centre, height, variance):
        """Initialise potential function class

        :param beta: float,  inverse temperature = 1 / (k_B * T)
        """
        self.dim = 2
        self.centre = centre
        self.height = height
        self.variance = variance

    def V(self, X):
        """Potential function

        :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)
        :return: V: float, potential energy value
        """
        assert (type(X) == np.ndarray)
        assert (X.ndim == 1)
        assert (X.shape[0] == 2)

        V = g(X, self.centre, self.height, self.variance)
        return V

    def dV_x(self, X):
        """
        :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)

        :return: dVx: float, differential of the potential with respect to x
        """

        return -(1/self.variance) * (X - self.centre)[0] * self.V(X)

    def dV_y(self, X):
        """
        :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)

        :return: dVy: float, differential of the potential with respect to x
        """

        return -(1/self.variance) * (X - self.centre)[1] * self.V(X)

    def nabla_V(self, X):
        """Gradient of potential fuction

        :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)
        :return: grad(X): np.array, gradient of position  vector (x,y), ndim = 1, shape = (2,)
        """
        assert (type(X) == np.ndarray)
        assert (X.ndim == 1)
        assert (X.shape[0] == 2)
        return np.array([self.dV_x(X), self.dV_y(X)])


class CombinePotential:
    """Class to combine potentials"""

    def __init__(self, beta, potentials, borne):
        """Initialise potential function class

        :param beta: float,  inverse temperature = 1 / (k_B * T)
        :param potentials: potentials, must have V, dVx, dVy, nabla_V methods
        :param borne: float, bornes on wich we integrate to find Z
        """
        self.beta = beta
        self.dim = 2
        self.potentials = potentials
        self.borne = borne
        self.Z = None
        self.set_Z()

    def V(self, X):
        """Potential fuction

        :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)
        :return: V: float, potential energy value
        """
        V = 0
        for potential in self.potentials:
            V += potential.V(X)
        return V

    def dV_x(self, X):
        """
        :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)
        :return: dVx: float, differential of the potential with respect to x
        """
        dVx = 0
        for potential in self.potentials:
            dVx += potential.dVx(X)
        return dVx

    def dV_y(self, X):
        """
        :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)
        :return: dVy: float, differential of the potential with respect to x
        """
        dVy = 0
        for potential in self.potentials:
            dVy += potential.dVy(X)
        return dVy

    def nabla_V(self, X):
        """Gradient of potential fuction

        :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)
        :return: grad(X): np.array, gradient of position  vector (x,y), ndim = 1, shape = (2,)
        """
        nabla_V = np.array([float(0),float(0)])
        for potential in self.potentials:
            nabla_V += potential.nabla_V(X)
        return nabla_V

    def set_Z(self):
        self.Z, _ = integrate.dblquad(self.boltz_weight, -self.borne, self.borne, -self.borne, self.borne)

    def boltz_weight(self, y, x):
        """Compute un normalized weight in the Botzmann distribution

        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: normalized Blotzmann weight
        """
        X = np.array([x, y])
        return np.exp(-self.beta * self.V(X))

    def create_potential_on_grid(self, precision):
        grid = np.linspace(-self.borne, self.borne, precision)
        potential_on_grid = np.zeros([precision, precision])
        for i in range(precision):
            for j in range(precision):
                potential_on_grid[i, j] = self.V(np.array([grid[i], grid[j]]))
        return grid, potential_on_grid


def create_circle_well_potential(beta, nb_well=8, height=-0.3, variance=0.4):

    centres = [(np.cos(2 * k * np.pi / nb_well), np.sin(2 * k * np.pi / nb_well)) for k in range(nb_well)]
    potentials = [WellPotential(centre, height, variance) for centre in centres]
    combined_potential = CombinePotential(beta, potentials, 1 + 3 * np.sqrt(variance))
    return combined_potential


def create_spiral_potential(beta, inter_space=3, r_min=0.6, r_max=3, height=-0.3, variance=0.3):
    delta_well = np.sqrt(2 * variance * np.log(2))

    r = r_min
    theta = 0
    centres = []
    while r < r_max:
        centres.append((r * np.cos(theta), r * np.sin(theta)))
        d_theta = delta_well / r
        d_r = inter_space * d_theta / (2*np.pi)
        r += d_r
        theta += d_theta

    potentials = [WellPotential(np.array(centre), height, variance) for centre in centres]
    combined_potential = CombinePotential(beta, potentials, r_max + 3 * np.sqrt(variance))
    return combined_potential

