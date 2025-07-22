import torch.nn as nn
import torch

class SnakeModelEmb(torch.nn.Module):

    def __init__(self):
        self.num_moteurs = 4 # Number of motors
        self.num_positions = 4 # Number of positions in the sequence
        self.vocab_size = 181 # Vocabulary size for the embedding (-90 to 90 degrees)
        self.embedding_size_moteur = 2 # Embedding size for each motor
        self.hidden_positions = 2 * self.num_moteurs * self.embedding_size_moteur # Hidden size for each position
        super(SnakeModelEmb, self).__init__()
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size_moteur, max_norm=1.0)
        self.positions = torch.nn.ModuleList([torch.nn.Linear(self.embedding_size_moteur * self.num_moteurs, self.hidden_positions) for _ in range(self.num_positions)])
        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.hidden_positions*self.num_positions, self.hidden_positions)
        self.linear2 = torch.nn.Linear(self.hidden_positions, self.embedding_size_moteur * self.num_moteurs)
        self.embedding_to_logits = torch.nn.Linear(self.embedding_size_moteur, self.vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [self.num_positions, self.num_moteurs, self.vocab_size]
        x = self.embedding(x)  # [self.num_positions, self.num_moteurs, self.embedding_size_moteur ]
        x = x.view(self.num_positions, -1)  # [self.num_positions, self.embedding_size_moteur  * self.num_moteurs]
        x = [self.activation(layer(x[i].unsqueeze(0))) for i, layer in enumerate(self.positions)]
        x = torch.cat(x, dim=1)  # [1, 100 * self.num_positions]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = x.view(-1, self.embedding_size_moteur)
        x = self.embedding_to_logits(x)
        x = self.softmax(x)
        return x
    
snakemodel = SnakeModelEmb()

print('The model:')
print(snakemodel)


# Generate a random input tensor for the model
# Shape: [num_positions, num_moteurs], values in [0, 180] (since embedding size is 181)
random_input = torch.randint(0, 181, (snakemodel.num_positions, snakemodel.num_moteurs))
print('\n\nRandom input:')
print(random_input)

# Pass the random input through the model
output = snakemodel(random_input)
# print('\n\nModel output shape:')
# print(output.shape)

# print('\n\nModel output:')
# print(output)

# Output to vocab index using top-p (nucleus) sampling
def top_p_sampling(logits, p=0.9):
    # logits: [num_moteurs, vocab_size]
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Create a mask for tokens to keep
    sorted_mask = cumulative_probs <= p
    # Always keep at least one token
    sorted_mask[..., 0] = True

    # Set logits of tokens outside top-p to a large negative value
    masked_logits = sorted_logits.masked_fill(~sorted_mask, float('-inf'))

    # Sample from the masked distribution
    probs = torch.softmax(masked_logits, dim=-1)
    sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Map back to original indices
    final_indices = sorted_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    return final_indices

# Use top-p sampling on the model output
vocab_indices = top_p_sampling(output, p=0.9)
print('\n\nSampled vocab indices (top-p):')
print(vocab_indices)