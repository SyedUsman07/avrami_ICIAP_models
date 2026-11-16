import torch

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        IN_DIM = 1 # temperature or time

        self.backbone = nn.Sequential(
            nn.Linear(IN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.d_head = nn.Linear(128, 1)
        self.Ea_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.A_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.n_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus())


    def forward(self, x):
        shared_feats = self.backbone(x)

        d = self.d_head(shared_feats)
        Ea = self.Ea_head(shared_feats)
        A = self.A_head(shared_feats)
        n = self.n_head(shared_feats)

        return d, Ea, A, n

class ModelRNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        IN_DIM = 1 # temperature or time
        # SEQ_LEN = 10 # length of the sequence

        self.backbone = nn.Sequential(
            nn.LSTM(IN_DIM, 32, num_layers=1, batch_first=True),  # input is [B, 10, 1], output is [B, 10, 32]
        )

        self.d_head = nn.Linear(32, 1)
        self.Ea_head = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        self.A_head = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        self.n_head = nn.Sequential(nn.Linear(32, 1), nn.Softplus())

    def forward(self, x):
        # x shape: [B, 10, 1, 1]
        x = x.view(x.size(0), -1, 1)
        x, _ = self.backbone(x)
        x = x[:, -1, :]

        # x shape: [B, 1, 32]
        d = self.d_head(x)
        Ea = self.Ea_head(x)
        A = self.A_head(x)
        n = self.n_head(x)
        # x shape: [B, 1, 1]
        return d, Ea, A, n



def __test__():
    net = ModelRNN()
    xin = torch.randn(2, 10, 1, 1)
    xout = net(xin)

if __name__ == "__main__":
    __test__()