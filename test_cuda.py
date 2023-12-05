import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Set random seed for reproducibility
torch.manual_seed(42)

# Define input size, hidden size, and output size
input_size = 10
hidden_size = 5
output_size = 1

# Instantiate the model
model = SimpleModel(input_size, hidden_size, output_size)

# Print the model architecture
print(model)

# Define a binary cross-entropy loss and an optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate random training data
num_samples = 100
X_train = torch.rand(num_samples, input_size)
y_train = torch.randint(0, 2, (num_samples, 1)).float()

# Training loop
num_epochs = 100000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)

    # Calculate the loss
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print the final trained model
print("Final trained model:")
print(model)
