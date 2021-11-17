#! /usr/bin/env python2.7

# --- for simulation
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
# --- for scientific computing ---
import numpy as np
from scipy import integrate
# --- for plots ---
import matplotlib.pyplot as plt
# --- for neural networks ---
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split as ttsplit

psf = CharmmPsfFile('A.psf')
pdb = PDBFile('A.pdb')
params = CharmmParameterSet('top_all27_prot_lipid.rtf', 'par_all27_prot_lipid.prm')

# System Configuration
nonbondedMethod = CutoffNonPeriodic
nonbondedCutoff = 1.4 * nanometers
constraints = HBonds
constraintTolerance = 0.00001

system = psf.createSystem(params, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
                          constraints=constraints)

# Integration Options
dt = 0.002 * picoseconds
temperature = 300.00 * kelvin
friction = 2 / picosecond
integrator = LangevinIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)

# do minimization, perform equilibration, then save the state of the simulation
equilibrationSteps = 25000
platform = Platform.getPlatformByName('CPU')
platformProperties = {'CpuThreads': '1'}

simulation = Simulation(psf.topology, system, integrator, platform, platformProperties)
simulation.context.setPositions(pdb.positions)

# Minimize and Equilibrate
print('Performing energy minimization...')
simulation.minimizeEnergy()
print('Equilibrating...')
simulation.context.setVelocitiesToTemperature(temperature)


def coordinates(name_file):
    array = np.zeros(66)
    file = open(name_file, 'r')
    for i in range(7):
        file.readline()
    potential=file.readline().split('"')[1]
    file.readline()
    for i in range(22):
        line = file.readline()
        line_split = line.split('"')
        array[3 * i] = float(line_split[1])
        array[3 * i + 1] = float(line_split[3])
        array[3 * i + 2] = float(line_split[5])
    return array, potential


traj = np.empty([equilibrationSteps, 66])
pot = np.empty(equilibrationSteps)


def set_learning_parameters(model, learning_rate, loss='MSE', optimizer='Adam'):
    """Function to set learning parameter

    :param model: Neural network model build with PyTorch,
    :param learning_rate: Value of the learning rate
    :param loss: String, type of loss desired ('MSE' by default, another choice leads to cross entropy)
    :param optimizer: String, type of optimizer ('Adam' by default, another choice leads to SGD)

    :return:
    """
    # --- chosen loss function ---
    if loss == 'MSE':
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.CrossEntropyLoss()
    # --- chosen optimizer ---
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return loss_function, optimizer


def train_AE(model, loss_function, optimizer, traj, weights, num_epochs=10, batch_size=32, test_size=0.2):
    """Function to train an AE model

    :param model: Neural network model built with PyTorch,
    :param loss_function: Function built with PyTorch tensors or built-in PyTorch loss function
    :param optimizer: PyTorch optimizer object
    :param traj: np.array, physical trajectory (in the potential pot), ndim == 2, shape == T // save + 1, pot.dim
    :param weights: np.array, weights of each point of the trajectory when the dynamics is biased, ndim == 1, shape == T // save + 1, 1
    :param num_epochs: int, number of times the training goes through the whole dataset
    :param batch_size: int, number of data points per batch for estimation of the gradient
    :param test_size: float, between 0 and 1, giving the proportion of points used to compute test loss

    :return: model, trained neural net model
    :return: training_data, list of lists of train losses and test losses; one per batch per epoch
    """
    # --- prepare the data ---
    # split the dataset into a training set (and its associated weights) and a test set
    X_train, X_test, w_train, w_test = ttsplit(traj, weights, test_size=test_size)
    X_train = torch.tensor(X_train.astype('float32'))
    X_test = torch.tensor(X_test.astype('float32'))
    w_train = torch.tensor(w_train.astype('float32'))
    w_test = torch.tensor(w_test.astype('float32'))
    # intialization of the methods to sample with replacement from the data points (needed since weights are present)
    train_sampler = torch.utils.data.WeightedRandomSampler(w_train, len(w_train))
    test_sampler = torch.utils.data.WeightedRandomSampler(w_test, len(w_test))
    # method to construct data batches and iterate over them
    train_loader = torch.utils.data.DataLoader(dataset=X_train,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=X_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=test_sampler)

    # --- start the training over the required number of epochs ---
    training_data = []
    for epoch in range(num_epochs):
        # Train the model by going through the whole dataset
        model.train()
        train_loss = []
        for iteration, X in enumerate(train_loader):
            # print(X)
            # Set gradient calculation capabilities
            X.requires_grad_()
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output
            out = model(X)
            # Evaluate loss
            loss = loss_function(out, X)
            # Store loss
            train_loss.append(loss)
            # Get gradient with respect to parameters of the model
            loss.backward()
            # Updating parameters
            optimizer.step()
        # Evaluate the test loss on the test dataset
        model.eval()
        with torch.no_grad():
            test_loss = []
            for iteration, X in enumerate(test_loader):
                out = model(X)
                # Evaluate loss
                loss = loss_function(out, X)
                # Store loss
                test_loss.append(loss)
            training_data.append([torch.tensor(train_loss), torch.tensor(test_loss)])
    return model, training_data


simulation.currentStep = 0
simulation.context.setTime(0.0)

for step in range(equilibrationSteps):
    simulation.step(1)
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    f = open('State.xml', 'w')
    f.write(XmlSerializer.serialize(state))
    f.close()
    traj[step],pot[step] = coordinates('State.xml')


class DeepAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, bottleneck_dim):
        super(DeepAutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[2]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[2], hidden_dims[3]),
            torch.nn.Tanh(),
            # torch.nn.Linear(hidden_dims[3], hidden_dims[4]),
            # torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[-1], bottleneck_dim),
            torch.nn.Tanh()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_dim, hidden_dims[-1]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[-1], hidden_dims[-2]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[-2], hidden_dims[-3]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[-3], hidden_dims[-4]),
            torch.nn.Tanh(),
            # torch.nn.Linear(hidden_dims[-4], hidden_dims[-5]),
            # torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[0], input_dim)
        )

    def forward(self, inp):
        encoded = self.encoder(inp)
        decoded = self.decoder(encoded)
        return decoded


# ---- parameters to change ----
batch_size = 100
num_epochs = 20
learning_rate = 0.005
ae1 = DeepAutoEncoder(66, [66, 50, 25, 10], 1)
print(ae1)

# --- training of the NN ---
loss_function, optimizer = set_learning_parameters(ae1, learning_rate=learning_rate)
(
    ae1,
    training_data1
) = train_AE(ae1,
             loss_function,
             optimizer,
             traj,
             np.ones(traj.shape[0]),
             batch_size=batch_size,
             num_epochs=num_epochs
             )

# --- Compute average losses per epoch ---
loss_evol1 = []
for i in range(len(training_data1)):
    loss_evol1.append([torch.mean(training_data1[i][0]), torch.mean(training_data1[i][1])])
loss_evol1 = np.array(loss_evol1)
print(loss_evol1)

# --- Plot the results ---
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
ax0.plot(loss_evol1[:, 0], '--', label='train loss', marker='x')
ax0.plot(range(1, len(loss_evol1[:, 1])), loss_evol1[: -1, 1], '-.', label='test loss', marker='+')
ax0.legend()
fig.show()

import sys
np.set_printoptions(threshold=sys.maxsize)

def pp(a):
    if (a<0.6):
        return 0
    else :
        return 1

def conditional_espectation(model):
    X = np.empty([len(traj), 66])
    Y = np.empty(len(traj))
    a1=0
    a2 = 0
    a3 = 0
    a4 = 0
    a5 = 0
    a6 = 0
    a7 = 0
    a8 = 0
    a9 = 0
    a10 = 0
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    b5 = 0
    b6 = 0
    b7 = 0
    b8 = 0
    b9 = 0
    b10 = 0
    for i in range(len(traj)):
        X[i] = traj[i]
        x = torch.tensor(X[i].astype('float32'))
        Y[i] = model.encoder(x)
        if (Y[i]<-0.8):
            a1+=1
            b1+=pp(X[i,0])
        elif (Y[i]<-0.6):
            a2+=1
            b2 += pp(X[i,0])
        elif (Y[i]<-0.4):
            a3+=1
            b3 += pp(X[i,0])
        elif (Y[i]<-0.2):
            a4+=1
            b4 += pp(X[i,0])
        elif (Y[i]<0):
            a5+=1
            b5 += pp(X[i,0])
        elif (Y[i]<0.2):
            a6+=1
            b6 += pp(X[i,0])
        elif (Y[i]<0.4):
            a7+=1
            b7 += pp(X[i,0])
        elif (Y[i]<0.6):
            a8+=1
            b8 += pp(X[i,0])
        elif (Y[i]<0.8):
            a9+=1
            b9 += pp(X[i,0])
        else :
            a10+=1
            b10 += pp(X[i,0])
    print(min(Y))
    print(max(Y))
    print("test 1")
    print(a1)
    print(a2)
    print(a3)
    print(a4)
    print(a5)
    print(a6)
    print(a7)
    print(a8)
    print(a9)
    print(a10)
    print("test 2")
    print(b1)
    print(b2)
    print(b3)
    print(b4)
    print(b5)
    print(b6)
    print(b7)
    print(b8)
    print(b9)
    print(b10)
    print("test 3")
    print(b1/a1)
    print(b2/a2)
    print(b3/a3)
    print(b4/a4)
    print(b5/a5)
    print(b6/a6)
    print(b7/a7)
    print(b8/a8)
    print(b9/a9)
    print(b10/a10)


conditional_espectation(ae1)
