import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class BCDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path) as f:
            d = json.load(f)
        self.obs = np.array(d['obs'], dtype=np.float32)
        self.action = np.array(d['action'], dtype=np.int64)

    def __len__(self):
        return len(self.action)

    def __getitem__(self, idx):
        return self.obs[idx], self.action[idx]


class PolicyNet(nn.Module):
    def __init__(self, input_dim=16, hidden=512, num_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='bc_dataset.json')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', type=str, default='bc_policy.pt')
    args = ap.parse_args()

    ds = BCDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PolicyNet().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for obs, act in dl:
            obs = obs.to(device)
            act = act.to(device)
            logits = model(obs)
            loss = crit(logits, act)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch+1}: loss {np.mean(losses):.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()


