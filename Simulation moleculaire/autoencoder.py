import numpy as np
# --- for plots ---
import matplotlib.pyplot as plt
# --- for neural networks ---
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split as ttsplit


class Training:
    def __init__(self, model, learning_rate, traj, weights, loss_function='MSE', optimizer='ADAM', num_epochs=10, batch_size=32, test_size=0.2):
        """
        :param model: Neural network model build with PyTorch,
        :param learning_rate: Value of the learning rate
        :param loss_function: String, type of loss desired ('MSE' by default, another choice leads to cross entropy)
        :param optimizer: String, type of optimizer ('Adam' by default, another choice leads to SGD)
        :param traj: np.array, physical trajectory (in the potential pot), ndim == 2, shape == T // save + 1, pot.dim
        :param weights: np.array, weights of each point of the trajectory when the dynamics is biased,
        ndim == 1, shape == T // save + 1, 1
        :param num_epochs: int, number of times the training goes through the whole dataset
        :param batch_size: int, number of data points per batch for estimation of the gradient
        :param test_size: float, between 0 and 1, giving the proportion of points used to compute test loss
        """
        self.model = model
        # --- chosen loss function ---
        if loss_function == 'MSE':
            self.loss_function = nn.MSELoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()
        # --- chosen optimizer ---
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.traj = traj
        self.weights = weights
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.test_size = test_size

    def train_AE(self):
        # --- prepare the data ---
        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test = ttsplit(self.traj, self.weights, test_size=self.test_size)
        X_train = torch.tensor(X_train.astype('float32'))
        X_test = torch.tensor(X_test.astype('float32'))
        w_train = torch.tensor(w_train.astype('float32'))
        w_test = torch.tensor(w_test.astype('float32'))
        # intialization of the methods to sample with replacement from the data points
        train_sampler = torch.utils.data.WeightedRandomSampler(w_train, len(w_train))
        test_sampler = torch.utils.data.WeightedRandomSampler(w_test, len(w_test))
        # method to construct data batches and iterate over them
        train_loader = torch.utils.data.DataLoader(dataset=X_train,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset=X_test,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  sampler=test_sampler)

        # --- start the training over the required number of epochs ---
        self.training_data = []
        for epoch in range(self.num_epochs):
            # Train the model by going through the whole dataset
            self.model.train()
            train_loss = []
            for iteration, X in enumerate(train_loader):
                # print(X)
                # Set gradient calculation capabilities
                X.requires_grad_()
                # Clear gradients w.r.t. parameters
                self.optimizer.zero_grad()
                # Forward pass to get output
                out = self.model(X)
                # Evaluate loss
                loss = self.loss_function(out, X)
                # Store loss
                train_loss.append(loss)
                # Get gradient with respect to parameters of the model
                loss.backward()
                # Updating parameters
                self.optimizer.step()
            # Evaluate the test loss on the test dataset
            self.model.eval()
            with torch.no_grad():
                test_loss = []
                for iteration, X in enumerate(test_loader):
                    out = self.model(X)
                    # Evaluate loss
                    loss = self.loss_function(out, X)
                    # Store loss
                    test_loss.append(loss)
                self.training_data.append([torch.tensor(train_loss), torch.tensor(test_loss)])
        return self.model

    def loss_evolution(self, plot=True):
        # --- Compute average losses per epoch ---
        loss_evol1 = []
        for i in range(len(self.training_data)):
            loss_evol1.append([torch.mean(self.training_data[i][0]), torch.mean(self.training_data[i][1])])
        loss_evol1 = np.array(loss_evol1)
        print(loss_evol1)

        # --- Plot the results ---
        if plot:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
            ax0.plot(loss_evol1[:, 0], '--', label='train loss', marker='x')
            ax0.plot(range(1, len(loss_evol1[:, 1])), loss_evol1[: -1, 1], '-.', label='test loss', marker='+')
            ax0.legend()
            fig.show()


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
