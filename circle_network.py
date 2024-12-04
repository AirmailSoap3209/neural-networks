import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define layers: input layer (2 features) -> hidden layer (8 neurons) -> output layer (1 neuron)
        self.layer1 = nn.Linear(2, 8)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)
        self.activation2 = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        return x

# Generate some sample data (circular pattern)
def generate_data(n_samples=1000):
    np.random.seed(0)
    
    # Generate random points
    X = np.random.randn(n_samples, 2)
    
    # Calculate distances from origin
    distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    
    # Assign labels: 1 if distance < 1, 0 otherwise
    y = (distances < 1).astype(np.float32)
    
    return X.astype(np.float32), y.reshape(-1, 1)

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Convert to torch tensor and predict for each point in the mesh
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = model(grid)
    Z = Z.numpy()
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap=plt.cm.RdYlBu)
    plt.xlabel('X₁')
    plt.ylabel('X₂')
    plt.title('Neural Network Decision Boundary')
    plt.colorbar(label='Prediction Probability')
    plt.show()

def main():
    # Generate data
    X, y = generate_data()
    
    # Convert to PyTorch tensors
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    
    # Create model
    model = SimpleNN()
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Lists to store loss history
    losses = []
    
    # Training loop
    print("Training the neural network...")
    for epoch in range(1000):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Store loss
        losses.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')
    
    # Plot training loss
    plt.figure(figsize=(10,6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Plot decision boundary
    plot_decision_boundary(model, X.numpy(), y.numpy())
    
    # Test the model with some sample points
    print("\nTesting the model with sample points:")
    test_points = torch.tensor([
        [0.5, 0.5],    # Should be inside the circle (close to 1)
        [2.0, 2.0],    # Should be outside the circle (close to 0)
        [0.1, 0.1],    # Should be inside the circle (close to 1)
        [1.5, -1.5]    # Should be outside the circle (close to 0)
    ], dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(test_points)
        for i, pred in enumerate(predictions):
            print(f"Point {test_points[i].numpy()}: {pred.item():.4f}")

if __name__ == "__main__":
    main()
