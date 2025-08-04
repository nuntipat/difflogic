import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from difflogic import LogicLayer, GroupSum
import einops
import time
import numpy as np
from tqdm import tqdm
from typing import Literal

# Type definitions
InitializationType = Literal['residual', 'random']

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 100
learning_rate = 0.01
num_epochs = 10
k = 16  # Base number of kernels (from paper: k=16 for small model)

print(f"Base kernel count k = {k}")
print(f"Expected shapes from paper:")
print(f"After conv1 + pool1: {k} × 12 × 12")
print(f"After conv2 + pool2: {3*k} × 6 × 6") 
print(f"After conv3 + pool3: {9*k} × 3 × 3")
print(f"After flattening: {81*k}")

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Logic gate definitions
logic_gates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def apply_logic_gate(a: torch.Tensor, b: torch.Tensor, logic_gate: int):
    return {
        0:  torch.zeros_like(a),
        1:  a * b,
        2:  a - a * b,
        3:  a,
        4:  b - a * b,
        5:  b,
        6:  a + b - 2 * a * b,
        7:  a + b - a * b,
        8:  1 - (a + b - a * b),
        9:  1 - (a + b - 2 * a * b),
        10: 1 - b,
        11: 1 - b + a * b,
        12: 1 - a,
        13: 1 - a + a * b,
        14: 1 - a * b,
        15: torch.ones_like(a),
    }[logic_gate]

class Logic(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 initialization_type: InitializationType = 'residual',
                 device=None
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.initialization_type = initialization_type
        self.device = device or torch.device('cpu')
        
        a, b = self.get_connections()
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        
        weights = torch.randn(out_dim, len(logic_gates), device=self.device)
        if self.initialization_type == 'residual':
            weights[:, :] = 0
            weights[:, 3] = 5  # Initialize to identity gate
        self.weights = torch.nn.parameter.Parameter(weights)

    def forward(self, x: torch.Tensor):
        a, b = x[:, self.a, ...], x[:, self.b, ...]
        
        if self.training:
            normalized_weights = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype).to(self.device)
            r = torch.zeros_like(a).to(x.dtype).to(self.device)
            for logic_gate in logic_gates:
                if len(a.shape) > 2:
                    nw = einops.repeat(normalized_weights[..., logic_gate], 'weights -> weights depth', depth=a.shape[-1])
                else:
                    nw = normalized_weights[..., logic_gate]
                r = r + nw * apply_logic_gate(a, b, logic_gate)
            return r
        else:
            one_hot_weights = torch.nn.functional.one_hot(self.weights.argmax(-1), len(logic_gates)).to(torch.float32).to(self.device)
            with torch.no_grad():
                r = torch.zeros_like(a).to(x.dtype).to(self.device)
                for logic_gate in logic_gates:
                    if len(a.shape) > 2:
                        ohw = einops.repeat(one_hot_weights[..., logic_gate], 'weights -> weights depth', depth=a.shape[-1])
                    else:
                        ohw = one_hot_weights[..., logic_gate]
                    r = r + ohw * apply_logic_gate(a, b, logic_gate)
                return r

    def get_connections(self):
        connections = torch.randperm(2 * self.out_dim) % self.in_dim
        connections = torch.randperm(self.in_dim)[connections]
        connections = connections.reshape(2, self.out_dim)
        a, b = connections[0], connections[1]
        a, b = a.to(torch.int64), b.to(torch.int64)
        a, b = a.to(self.device), b.to(self.device)
        return a, b

class LogicTree(nn.Module):
    def __init__(self,
                 in_dim: int,
                 depth: int = 3,
                 initialization_type: InitializationType = 'residual',
                 device=None,
                 ):
        super().__init__()
        self.device = device or torch.device('cpu')
        
        layers = [LogicLayer(in_dim, int(2 ** (depth - 1)), initialization_type=initialization_type, device=self.device,implementation='cuda' if device.type == 'cuda' else 'python',
                    connections='random',grad_factor=1.5 )]
        for i in range(0, depth - 1, 1):
            layers.append(LogicLayer(int(2 ** (depth - 1 - i)), int(2 ** (depth - 1 - i - 1)), 
                            initialization_type=initialization_type, device=self.device,implementation='cuda' if device.type == 'cuda' else 'python',
                            connections='random',grad_factor=1.5))
        
        self.tree = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.tree(x)

class Conv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth: int = 3,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 initialization_type: InitializationType = 'residual',
                 device=None
                 ):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.device = device or torch.device('cpu')
        
        self.filters = nn.ModuleList([
            LogicTree(in_dim=kernel_size ** 2 * in_channels, depth=depth, 
                     initialization_type=initialization_type, device=self.device) 
            for _ in range(out_channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        outputs = []
        
        patches = einops.rearrange(patches, 'b h w -> (b w) h', h=patches.shape[1], w=patches.shape[2]) # Input is (100,25,576) Output: (57600,25)
        for filter in self.filters:
            out = filter(patches)  # Shape: (batch_size, 1, out_height * out_width)
            out = einops.rearrange(out, '(b h w) 1 -> b (h w)', h=out_height, w=out_width)
            outputs.append(out)
        
        output_tensor = torch.stack(outputs, dim=1)  # Shape: (batch_size, out_channels, out_height * out_width)
        output_tensor = einops.rearrange(output_tensor, 'b c (h w) -> b c h w', h=out_height, w=out_width)
        return output_tensor

class CustomOrPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(CustomOrPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        # Use MaxPool2d as approximation to OR pooling
        # In binary logic, max operation approximates OR
        return torch.max_pool2d(x, self.kernel_size, self.stride, self.padding)



class ConvDiffLogicMNIST(nn.Module):
    def __init__(self, k=16):
        super(ConvDiffLogicMNIST, self).__init__()
        self.k = k
        
        # Convolutional block 1: k kernels, 5x5, depth=3, no padding
        # Input: 1 × 28 × 28 -> Output: k × 24 × 24 (28-5+1=24)
        self.conv1 = Conv(in_channels=1, out_channels=k, kernel_size=5, depth=3, 
                         padding=0, initialization_type='residual', device=device)
        
        # OR pooling 1: 2x2, stride 2
        # k × 24 × 24 -> k × 12 × 12
        self.pool1 = CustomOrPool2d(kernel_size=2, stride=2)
        
        # Convolutional block 2: 3*k kernels, 3x3, depth=3
        # k × 12 × 12 -> 3*k × 12 × 12 (with padding=1), then pooled to 3*k × 6 × 6
        self.conv2 = Conv(in_channels=k, out_channels=3*k, kernel_size=3, depth=3, 
                         padding=1, initialization_type='residual', device=device)
        
        # OR pooling 2: 2x2, stride 2
        # 3*k × 12 × 12 -> 3*k × 6 × 6
        self.pool2 = CustomOrPool2d(kernel_size=2, stride=2)
        
        # Convolutional block 3: 9*k kernels, 3x3, depth=3
        # 3*k × 6 × 6 -> 9*k × 6 × 6 (with padding=1), then pooled to 9*k × 3 × 3
        self.conv3 = Conv(in_channels=3*k, out_channels=9*k, kernel_size=3, depth=3, 
                         padding=1, initialization_type='residual', device=device)
        
        # OR pooling 3: 2x2, stride 2
        # 9*k × 6 × 6 -> 9*k × 3 × 3
        self.pool3 = CustomOrPool2d(kernel_size=2, stride=2)
        
        # Flatten: 9*k × 3 × 3 -> 81*k
        self.flatten = nn.Flatten()
        
        # Regular differentiable logic layers (as specified in paper)
        # 81*k → 1280*k
        self.fc1 = LogicLayer(
            in_dim=81*k,
            out_dim=1280*k,
            device=device,
            implementation='cuda' if device.type == 'cuda' else 'python',
            connections='random',
            grad_factor=1.5 , # Higher for deeper networks
        )
        
        # 1280*k → 640*k
        self.fc2 = LogicLayer(
            in_dim=1280*k,
            out_dim=640*k,
            device=device,
            implementation='cuda' if device.type == 'cuda' else 'python',
            connections='random',
            grad_factor=1.5
        )
        
        # 640*k → 320*k
        self.fc3 = LogicLayer(
            in_dim=640*k,
            out_dim=320*k,
            device=device,
            implementation='cuda' if device.type == 'cuda' else 'python',
            connections='random',
            grad_factor=1.5
        )
        
        # GroupSum: 320*k → 10 (10 classes)
        # Using tau=30 as in the paper specifications
        self.group_sum = GroupSum(k=10, tau=30)
        
    def forward(self, x):
        # Input thresholding for binary processing (as mentioned in paper)
        # The paper mentions using binary inputs
        x = (x > 0.5).float()
        
        # Debug shape printing (uncomment for debugging)
        # print(f"Input shape: {x.shape}")
        
        # Convolutional processing with logic gates
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        
        x = self.pool1(x)
        # print(f"After pool1: {x.shape}")
        
        x = self.conv2(x)
        # print(f"After conv2: {x.shape}")
        
        x = self.pool2(x)
        # print(f"After pool2: {x.shape}")
        
        x = self.conv3(x)
        # print(f"After conv3: {x.shape}")
        
        x = self.pool3(x)
        # print(f"After pool3: {x.shape}")
        
        # Flatten
        x = self.flatten(x)
        # print(f"After flatten: {x.shape}")
        
        # Fully connected logic layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        # GroupSum for classification
        x = self.group_sum(x)
        
        return x

# Initialize model
model = ConvDiffLogicMNIST(k=k).to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Print architecture details
print("\n" + "="*80)
print("LOGIC GATE CONVOLUTIONAL DIFFLOGIC MNIST")
print("="*80)
print(f"Input: 1 × 28 × 28")
print(f"Conv1: {k} logic gate filters, 5×5, depth=3, no padding -> {k} × 24 × 24")
print(f"Pool1: OR pooling 2×2, stride 2 -> {k} × 12 × 12")
print(f"Conv2: {3*k} logic gate filters, 3×3, depth=3 -> {3*k} × 12 × 12")
print(f"Pool2: OR pooling 2×2, stride 2 -> {3*k} × 6 × 6")
print(f"Conv3: {9*k} logic gate filters, 3×3, depth=3 -> {9*k} × 6 × 6")
print(f"Pool3: OR pooling 2×2, stride 2 -> {9*k} × 3 × 3")
print(f"Flatten: -> {81*k}")
print(f"FC1: Regular differentiable logic layer {81*k} -> {1280*k}")
print(f"FC2: Regular differentiable logic layer {1280*k} -> {640*k}")
print(f"FC3: Regular differentiable logic layer {640*k} -> {320*k}")
print(f"GroupSum: {320*k} -> 10 classes")
print("="*80)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing')
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            progress_bar.set_postfix({
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    return test_loss / len(test_loader), 100. * correct / total

if __name__ == '__main__':
    print("Starting training with logic gate convolutions...")
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_logic_conv_difflogic_mnist.pth')
            print(f"New best model saved! Test accuracy: {test_acc:.2f}%")
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Best test accuracy: {best_test_acc:.2f}%")

    # Test discrete inference (switch to hard logic gates)
    print("\nTesting discrete inference...")
    model.eval()  # This switches to discrete/hard logic mode

    start_time = time.time()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    inference_time = time.time() - start_time
    inference_speed = inference_time / total

    print(f"Discrete inference accuracy: {100. * correct / total:.2f}%")
    print(f"Inference speed: {inference_speed:.6f} seconds per sample")

    # Save final model
    torch.save(model.state_dict(), 'final_logic_conv_difflogic_mnist.pth')

    # Final summary
    print("\n" + "="*80)
    print("LOGIC GATE CONVOLUTIONAL DIFFLOGIC MNIST RESULTS")
    print("="*80)
    print(f"Architecture: Custom logic gate convolutions + DiffLogic FC layers")
    print(f"Base kernel count (k): {k}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training epochs: {num_epochs}")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Final test accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Inference speed: {inference_speed:.6f} seconds per sample")
    print()
    print("Logic gate convolutions replace traditional conv layers with:")
    print("- 16 different logic gates (AND, OR, XOR, NOT, etc.)")
    print("- Tree-structured logic processing with configurable depth")
    print("- Soft logic during training, hard logic during inference")
    print("="*80)