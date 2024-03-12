import torch

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create two random matrices

matrix1 = torch.rand(10000, 10000).to(device)
matrix2 = torch.rand(10000, 10000).to(device)

# Multiply the matrices
for i in range(100):
    result = torch.matmul(matrix1, matrix2)

# Wait for a key press before exiting
input("Press Enter to exit...")