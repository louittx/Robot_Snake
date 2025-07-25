import json
from modelEmb import SnakeModelEmb
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim


class SnakeDataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            raw_data = json.load(f)

        self.samples = []
        for example in raw_data:
            if example["moved"]:  # Garde uniquement les exemples "moved == True"
                positions = torch.tensor(example["positions"], dtype=torch.long)        # [4, 4]
                next_position = torch.tensor(example["next_position"], dtype=torch.long)  # [4]
                self.samples.append((positions, next_position))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def compute_loss(logits, targets, criterion):
    """
    logits: [num_moteurs, vocab_size]
    targets: [num_moteurs]
    """
    losses = [criterion(logits[i], targets[i]) for i in range(targets.size(0))]
    return torch.stack(losses).mean()




def train_snake_model(model, dataset_path, num_epochs=10, batch_size=1, lr=1e-3):
    dataset = SnakeDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch  # inputs: [B, 4, 4], targets: [B, 4]
            inputs = inputs.squeeze(0)  # [4, 4]
            targets = targets.squeeze(0)  # [4]

            optimizer.zero_grad()
            logits = model(inputs)  # [4, 181]
            loss = compute_loss(logits, targets, criterion)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    return model


model = SnakeModelEmb()
trained_model = train_snake_model(model, "/Users/xblanc/devs/Robot_Snake/SnakeMujoco/snake_dataset_10000.json")


# Sauvegarde
torch.save(trained_model.state_dict(), "snake_model.pt")

# Chargement
model = SnakeModelEmb()
model.load_state_dict(torch.load("snake_model.pt"))
model.eval()
