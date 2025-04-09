import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ggg

# Transformation for MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create model, loss function, and optimizer
model = ggg.KANetwork(hidden_layers=[784, 128, 64, 10])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.view(-1, 784))  # Flatten MNIST images to 1D tensor
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / len(train_loader.dataset) * 100:.2f}%')

# Train the model
train_model(model, train_loader, criterion, optimizer, device)
