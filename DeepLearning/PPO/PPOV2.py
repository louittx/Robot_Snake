import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# === Réseau pour l'acteur (policy) ===
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 16)
        self.out = nn.Linear(16, 4)  # 4 actions

    def forward(self, x):
        x = F.relu(self.fc(x))
        probs = F.softmax(self.out(x), dim=-1)
        return probs

# === Réseau pour le critique (valeur estimée) ===
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        value = self.out(x)
        return value

# === Initialisation ===
policy_net = PolicyNetwork()
value_net = ValueNetwork()
optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.01)
optimizer_value = optim.Adam(value_net.parameters(), lr=0.01)

# === Mémoire de trajectoire ===
states = []
actions = []
log_probs = []
rewards = []
values = []

# === Une boucle d'entraînement très simple ===
for episode in range(5):  # 5 interactions manuelles
    print(f"\n--- Épisode {episode+1} ---")
    state = torch.rand(4)  # vecteur d'entrée aléatoire
    probs = policy_net(state)
    value = value_net(state)

    # échantillonner une action selon la distribution
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)

    print("État :", state.tolist())
    print("Probabilités des actions :", probs.tolist())
    print("Action choisie :", action.item())

    # récompense donnée manuellement
    reward = float(input("Résultat ? (1 pour True / 0 pour False) : "))

    # stocker pour mise à jour plus tard
    states.append(state)
    actions.append(action)
    log_probs.append(log_prob)
    rewards.append(reward)
    values.append(value)

# === Mise à jour PPO après les épisodes ===
print("\n--- Mise à jour des poids ---")
states = torch.stack(states)
actions = torch.tensor(actions)
log_probs = torch.stack(log_probs)
values = torch.stack(values).squeeze()
rewards = torch.tensor(rewards)

# calcul des avantages
advantages = rewards - values.detach()

# === Loss Policy (PPO simplifié sans clip) ===
policy_loss = - (log_probs * advantages).mean()

# === Loss Critic ===
value_loss = F.mse_loss(values, rewards)

# rétropropagation
optimizer_policy.zero_grad()
policy_loss.backward()
optimizer_policy.step()

optimizer_value.zero_grad()
value_loss.backward()
optimizer_value.step()

print("Loss policy :", policy_loss.item())
print("Loss critic :", value_loss.item())
