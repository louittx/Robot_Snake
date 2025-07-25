import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Définition de la "politique" et "valeur" avec 1 seul neurone
class SimplePolicy:
    def __init__(self):
        self.W = torch.randn(1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        return self.W * x + self.b

    def get_action(self, x):
        mean = self.forward(x)
        std = torch.tensor(0.5)  # constante ici
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, dist

class SimpleValue:
    def __init__(self):
        self.W = torch.randn(1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        return self.W * x + self.b

# Optimisateurs partagés
params_policy = []
params_value = []

# Instanciation
policy = SimplePolicy()
value = SimpleValue()

params_policy = [policy.W, policy.b]
params_value = [value.W, value.b]

optimizer_policy = torch.optim.Adam(params_policy, lr=1e-2)
optimizer_value = torch.optim.Adam(params_value, lr=1e-2)

# Simulation PPO simplifiée
gamma = 0.99
eps_clip = 0.2
epochs = 100000
x = torch.tensor([1.0])

losses = []

for epoch in tqdm(range(epochs)):
    # Collecte des données
    actions = []
    log_probs_old = []
    rewards = []
    values = []

    for _ in range(5):  # 5 épisodes par update
        action, log_prob, dist = policy.get_action(x)
        reward = -abs(action)  # on veut se rapprocher de 0
        v = value.forward(x)

        actions.append(torch.tensor(action))
        log_probs_old.append(log_prob.detach())
        rewards.append(torch.tensor(reward))
        values.append(v)

    # Calcul du retour (ici juste R = r car pas d'environnement)
    returns = rewards
    values = torch.stack(values)
    returns = torch.stack(returns).detach()

    # Avantage estimé : A = R - V
    advantages = returns - values.detach()

    # Mise à jour de la valeur (critic)
    value_loss = F.mse_loss(values.squeeze(), returns)
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    # Mise à jour de la politique (actor) avec PPO
    new_log_probs = []
    for a in actions:
        mean = policy.forward(x)
        std = torch.tensor(0.5)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs.append(dist.log_prob(a))

    new_log_probs = torch.stack(new_log_probs)
    log_probs_old = torch.stack(log_probs_old)
    ratios = torch.exp(new_log_probs - log_probs_old)

    surrogate1 = ratios * advantages
    surrogate2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
    policy_loss = -torch.min(surrogate1, surrogate2).mean()

    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    losses.append(policy_loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {policy_loss.item():.4f} - Action moyenne: {sum(actions)/len(actions):.4f}")

# Tracé de l'évolution de la loss
plt.plot(losses)
plt.title("Loss PPO")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
