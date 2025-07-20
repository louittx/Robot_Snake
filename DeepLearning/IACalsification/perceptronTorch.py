import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

def init(X):
    W = torch.randn((X.shape[1], 1), dtype=torch.float32, requires_grad=True)
    b = torch.randn((1,), dtype=torch.float32, requires_grad=True)
    return W, b

def model(X, W, b):
    Z = X @ W + b
    A = torch.sigmoid(Z)
    return A

def cost(A, y):
    return (-1 / len(y)) * torch.sum(y * torch.log(A + 1e-8) + (1 - y) * torch.log(1 - A + 1e-8))

def predict(X, W, b):
    with torch.no_grad():
        A = model(X, W, b)
        return (A >= 0.5).float()

def algo(X_np, y_np, lr=0.1, iterations=100):
    # Convert numpy arrays to torch tensors
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    W, b = init(X)
    log_loss = []

    for i in range(iterations):
        A = model(X, W, b)
        loss = cost(A, y)
        log_loss.append(loss.item())

        # Backpropagation
        loss.backward()

        # Gradient descent update
        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad

        # Zero gradients
        W.grad.zero_()
        b.grad.zero_()

    y_pred = predict(X, W, b)
    acc = accuracy_score(y_np, y_pred.numpy())
    print("Accuracy:", acc)
    return W, b, log_loss

# Dataset
X_np, y_np = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y_np = y_np.reshape(-1, 1)

# Training
W, b, losses = algo(X_np, y_np, lr=0.1, iterations=200)

# Affichage
x0 = torch.linspace(-1, 4, 100)
x1 = (-W[0] * x0 - b) / W[1]
x1 = x1.detach().numpy()
x0 = x0.detach().numpy()

plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np[:, 0], cmap="summer")
plt.plot(x0, x1, c="orange", lw=3)
plt.title("Séparation par régression logistique")
plt.show()

# Nouvelle prédiction
NewData = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
value = predict(NewData, W, b)
print("Prediction:", value.item())
