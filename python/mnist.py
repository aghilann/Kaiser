import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from typing import Dict, Tuple, List

def task(container_image: str = None):
    def decorator(func):
        func.container_image = container_image
        return func
    return decorator

def workflow(func):
    return func

@task(container_image="intel/deep-learning:2024.2-py3.10")
def load_data() -> datasets.MNIST:
    from torchvision import datasets, transforms
    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return dataset

@task(container_image="intel/deep-learning:2024.2-py3.10")
def preprocess_data(dataset: datasets.MNIST) -> Tuple[DataLoader, DataLoader]:
    from torch.utils.data import DataLoader, random_split
    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader

@task(container_image="intel/deep-learning:2024.2-py3.10")
def train_model(train_loader: DataLoader) -> Dict[str, torch.Tensor]:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(2):  # 5 epochs for demonstration
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                      f'\tLoss: {loss.item():.6f}')

    return model.state_dict()

@task(container_image="intel/deep-learning:2024.2-py3.10")
def evaluate_model(state_dict: Dict[str, torch.Tensor], test_loader: DataLoader) -> float:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    model = Net()
    model.load_state_dict(state_dict)  # Load the trained parameters
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}')
    return accuracy

@workflow
def mnist_workflow() -> None:
    dataset = load_data()
    train_loader, test_loader = preprocess_data(dataset)
    model = train_model(train_loader)
    accuracy = evaluate_model(model, test_loader)

if __name__ == "__main__":
    mnist_workflow()