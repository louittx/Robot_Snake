import torch
import torch.nn as nn
from tqdm import tqdm

class Reseau(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.net(x)

Essai = 10000
policy = Reseau()
optimiseur = torch.optim.Adam(policy.parameters(), lr=1e-3)

entrée = torch.randn(1, 3)
for i in tqdm(range(Essai)):
    sortie = policy(entrée)

    # Supposons que la cible soit [0.0, 1.0]
    cible = torch.tensor([[0.0, 140.0]])
    perte = nn.MSELoss()(sortie, cible)

    # Rétropropagation
    optimiseur.zero_grad()
    perte.backward()
    optimiseur.step()

print("Sortie : ",sortie)
