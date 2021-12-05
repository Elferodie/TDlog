from simulation_openmm import *
from autoencoder import *

simulation = Simulationn("Fichiers_alanine-dipeptide/A.psf","Fichiers_alanine-dipeptide/A.pdb",22)
traj, pot = simulation.trajectory()

training = Training(model = DeepAutoEncoder(66, [66, 50, 25, 10], 1), learning_rate=0.005, traj=traj, weights=np.ones(traj.shape[0]), num_epochs=20, batch_size=100,
                 test_size=0.2)
training.train_AE()
training.loss_evolution(False)