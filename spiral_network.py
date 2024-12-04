import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ComplexNN(nn.Module):
    def __init__(self):
        super(ComplexNN, self).__init__()
        # Much deeper network with more neurons
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # Batch normalization for better training
            nn.Dropout(0.2),     # Dropout for regularization
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def generate_spiral_data(n_points=1000, n_classes=2):
    """Generate spiral data for binary classification"""
    points_per_class = n_points // n_classes
    X = np.zeros((n_points, 2))
    y = np.zeros(n_points)
    
    for class_idx in range(n_classes):
        ix = range(points_per_class * class_idx, points_per_class * (class_idx + 1))
        r = np.linspace(0.0, 1, points_per_class)  # radius
        t = np.linspace(class_idx * 4, (class_idx + 1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_idx

    return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)

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
    plt.title('Neural Network Decision Boundary - Spiral Pattern')
    plt.colorbar(label='Prediction Probability')
    plt.show()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate spiral data
    X, y = generate_spiral_data(n_points=2000)
    
    # Convert to PyTorch tensors
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    
    # Create model
    model = ComplexNN()
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Lists to store loss history
    losses = []
    
    # Training loop
    print("Training the neural network...")
    n_epochs = 2000  # More epochs for complex data
    
    for epoch in range(n_epochs):
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
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
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
        [0.5, 0.5],
        [-0.5, -0.5],
        [0.8, -0.2],
        [-0.2, 0.8]
    ], dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(test_points)
        for i, pred in enumerate(predictions):
            print(f"Point {test_points[i].numpy()}: {pred.item():.4f}")

if __name__ == "__main__":
    main()
