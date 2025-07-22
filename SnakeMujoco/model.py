import torch.nn as nn
import torch

class SnakeModel(torch.nn.Module):

    def __init__(self):
        super(SnakeModel, self).__init__()
        self.num_moteurs = 4 # Number of motors
        self.num_positions = 4 # Number of positions in the sequence
        self.hidden_layer_size = 2 * self.num_moteurs * self.num_positions # Hidden size for each position
        self.hidden_1 = torch.nn.Linear(self.num_moteurs*self.num_positions, self.hidden_layer_size)
        self.hidden_2 = torch.nn.Linear(self.hidden_layer_size, self.num_moteurs)
        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.hidden_1(x)
        x = self.activation(x)
        x = self.hidden_2(x)
        # x = self.activation(x)
        x = torch.clamp(x, -90, 90)
        return x
    
snakemodel = SnakeModel()

print('The model:')
print(snakemodel)


# Generate a random input tensor for the model
# Shape: [num_positions, num_moteurs], values in [-90, 90] 
random_input = torch.randint(-90, 91, (4, 4)).float()  # 4 positions, 4 motors
random_input = random_input.view(1, -1)  # Reshape to [1, num_positions * num_moteurs]
print('\n\nRandom input shape:')
print(random_input.shape)
print('Random input values:')
print(random_input)

# Pass the random input through the model
output = snakemodel(random_input)
# print('\n\nModel output shape:')
# print(output.shape)

print('\n\nModel output:')
print(output)
