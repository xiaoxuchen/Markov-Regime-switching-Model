import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time
import datetime

#%% Define the dataset
class Bus_data(Dataset):
    def __init__(self, data):
        # data in the shape of (num_samples, seq_len * 2)
        self.seq_len = data.shape[1] // 2
        self.travel_time = torch.tensor(data[:, :self.seq_len], dtype=torch.float32)
        self.occupancy = torch.tensor(data[:, self.seq_len:], dtype=torch.float32)

    def __len__(self):
        return self.travel_time.size(0)

    def __getitem__(self, idx):
        travel_time = self.travel_time[idx]  # shape (seq_len,)
        occupancy = self.occupancy[idx]  # shape (seq_len,)

        x = torch.stack([travel_time, occupancy], dim=1)  # shape (seq_len, 2)
        features = torch.arange(self.seq_len-1)  # shape (seq_len - 1,), represents the position of the link
        y = x[1:]  # shape (seq_len - 1, 2)
        return (x[:-1], features), y

#%% Define the DeepAR model
class NormalHead(nn.Module):
    """Produces parameters for a normal distribution, log-Normal distribution, or truncated normal distribution."""
    def __init__(self, d_model, output_len, dropout=0, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mu = nn.Linear(d_model, output_len)
        self.sigma = nn.Linear(d_model, output_len)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: [bs x num_patch x d_model]
        mu = self.mu(self.dropout(x))
        sigma = self.softplus(self.sigma(self.dropout(x)))
        return mu, sigma

class SampleNormal:
    def __init__(self, **kwargs):
        pass
    def __call__(self, mu, sigma):
        dist = torch.distributions.Normal(mu, sigma)
        return dist.sample((1,)).squeeze(0)

class MeanNormal:
    def __init__(self, **kwargs):
        pass
    def __call__(self, mu, sigma):
        return mu


NORMAL_SCALE = np.log(2 * np.pi)*0.5
def nll_normal(mu, sigma, y):
    """Compute the negative log likelihood of Normal distribution, element-wise."""
    nnl = torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + NORMAL_SCALE
    return nnl

def normal_loss(mu, sigma, y, **kwargs):
    return torch.mean(nll_normal(mu, sigma, y))

class DeepAR(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, dropout):
        super(DeepAR, self).__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.head = NormalHead(hidden_size, input_size, dropout)
        self.mean = MeanNormal()
        self.sample = SampleNormal()
        self.pos_embed = nn.Embedding(seq_len, hidden_size)

    def forward(self, x, state=None, method='param'):
        '''
        Args:
            x: [batch_size, seq_length, 2] + a list of features of shape [batch_size, seq_length]
            state: initial state of (hidden, cell), default None means (zeros)
            method: 'param','mean' or 'sample'
        '''
        z = x[0]
        features = x[1]
        z = self.proj(z) + self.pos_embed(features)

        # LSTM
        if state is None:
            z, (hidden, cell) = self.lstm(z)
        else:
            hidden, cell = state
            z, (hidden, cell) = self.lstm(z, (hidden, cell))  # z: [bs x num_patch x d_model]

        # Head
        z = self.head(z)

        if method=='param':
            return z
        elif method=='mean':
            return self.mean(*z), (hidden, cell)
        elif method=='sample':
            return self.sample(*z), (hidden, cell)
        else:
            raise ValueError("method should be 'param', 'mean', or 'sample'.")


    def forecast(self, x, features, method='mean'):
        """Autoregressive forecasting in the test phase
        x: [batch_size, length, 2] + a list of features of shape [batch_size, seq_length]
        features: [batch_size, seq_length]
        method: 'mean' or 'sample'
        """
        n_target = features.shape[1] - x.shape[1] + 1
        n_input = x.shape[1]

        result = []
        self.eval()
        with torch.no_grad():
            xx = (x, features[:, :n_input])
            y_new, state = self(xx, method=method)
            for i in range(n_target-1):
                y_new = y_new[:, [-1], :]
                result.append(y_new)
                xx = (y_new, features[:, [n_input + i]])
                y_new, state = self(xx, state, method=method)
            result.append(y_new)
            result = torch.cat(result, dim=1)

        return result

    def forecast_samples(self, x, features, n=100):
        """Autoregressive forecasting in the test phase, draw n samples
        """
        with torch.no_grad():
            if x.shape[0] == 1:
                xx = x.repeat(n, 1, 1)
                features = features.repeat(n, 1)
                result = self.forecast(x=xx, features=features, method='sample')
                return result
            else:
                result = []
                with torch.no_grad():
                    for i in range(n):
                        result.append(self.forecast(x=x, features=features, method='sample'))
                return torch.stack(result, dim=0)


def train_model(train_loader, val_loader, model, n_epochs=100, lr=1e-3, patience=7):
    trian_start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters())

    # learning rate scheduler, onecyle policy
    step_loss = []
    epoch_loss = []
    best_val_loss = np.inf
    epochs_no_improve = 0
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
                                                epochs=n_epochs, pct_start=5/n_epochs,
                                                div_factor=1000, final_div_factor=10000,
                                                anneal_strategy='linear')

    for epoch in range(n_epochs):
        model.train()
        for i, (inputs, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(inputs)
            loss = normal_loss(*output, target)
            step_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()


        epoch_loss.append(np.mean(step_loss[-len(train_loader):]))

        # Calculate validation loss
        model.eval()
        val_loss = []
        for i, (inputs, target) in enumerate(val_loader):
            output = model(inputs)
            loss = normal_loss(*output, target)
            val_loss.append(loss.item())
        print(f'Epoch [{epoch}/{n_epochs}], Val Loss: {np.mean(val_loss):.4f} \t Train Loss: {epoch_loss[-1]:.4f} '
              f'\t total time: {time.time() - trian_start_time:.2f}')
        best_val_loss = min(best_val_loss, np.mean(val_loss))

        # Save the current best model
        if np.mean(val_loss) == best_val_loss:
            torch.save(model.state_dict(), 'deepar.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print('Early stopping!')
            break
        if time.time() - trian_start_time > 1*3600:
            print(f'Time limit {1} hours reached! Stopping training.')
            break

    # Load the best model
    model.load_state_dict(torch.load(f'deepar.pth'))

    return model







