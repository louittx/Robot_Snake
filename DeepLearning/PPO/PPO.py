import numpy as np

# Paramètres de la politique (1 seule entrée / sortie pour simplifier)
w_mu = np.random.randn()
b_mu = np.random.randn()

# Paramètres de la valeur
w_v = np.random.randn()
b_v = np.random.randn()

# Hyperparamètres
epsilon = 0.2
gamma = 0.99
lr = 1e-2

def policy_mean(s):
    return w_mu * s + b_mu

def value(s):
    return w_v * s + b_v

def policy_prob(mu, a, sigma=1.0):
    # Densité gaussienne : proba de a donné mu
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((a - mu)/sigma)**2)


def generate_episode():
    states, actions, rewards = [], [], []
    s = np.random.uniform(-1, 1)  # état initial
    for _ in range(5):  # épisode court
        mu = policy_mean(s)
        a = np.random.normal(mu, 1.0)
        r = -abs(a)  # récompense = -|a| pour encourager a → 0

        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = np.random.uniform(-1, 1)  # nouvel état aléatoire
    return states, actions, rewards


def compute_returns(rewards):
    R, returns = 0, []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def compute_advantages(states, returns):
    return [R - value(s) for s, R in zip(states, returns)]


def update(states, actions, advantages, old_probs):
    global w_mu, b_mu, w_v, b_v
    dw_mu, db_mu = 0, 0
    dw_v, db_v = 0, 0

    for s, a, A, old_p in zip(states, actions, advantages, old_probs):
        mu = policy_mean(s)
        p = policy_prob(mu, a)
        r = p / (old_p + 1e-8)

        clipped_r = np.clip(r, 1 - epsilon, 1 + epsilon)
        loss_grad = -min(r * A, clipped_r * A)

        # Gradient du mu par rapport à w_mu, b_mu : mu = w*s + b
        dmu = -A * (1 if r < clipped_r else 0) * (a - mu)
        dw_mu += dmu * s
        db_mu += dmu

        # Valeur (loss simple MSE)
        v = value(s)
        dv = 2 * (v - (A + v))  # perte MSE entre v et retour
        dw_v += dv * s
        db_v += dv

    # Mise à jour des poids
    w_mu -= lr * dw_mu
    b_mu -= lr * db_mu
    w_v -= lr * dw_v
    b_v -= lr * db_v


for episode in range(1000):
    states, actions, rewards = generate_episode()
    returns = compute_returns(rewards)
    advantages = compute_advantages(states, returns)

    old_probs = [policy_prob(policy_mean(s), a) for s, a in zip(states, actions)]

    update(states, actions, advantages, old_probs)

    if episode % 100 == 0:
        print("Episode", episode, "Total reward:", sum(rewards))
