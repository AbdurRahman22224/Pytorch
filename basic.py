import torch 

x = torch.tensor([1, 2, 3], dtype = torch.float32)
y = torch.tensor([3, 5, 6], dtype = torch.float32)

print(torch.cuda.is_available())

# Addition 
z = x + y 
z = torch.add(x, y)
print(z)

# Subtraction

# z = x - y
z = torch.subtract(x, y)
print(z)

# division    
z = x / y
d = torch.divide(x, y, rounding_mode = 'floor')
z = torch.true_divide(x, y)      # rounding_mode = None
print('Division: ', d)
print('True Division: ', z)

# inplace operations
t = torch.empty(3)
t.add_(torch.tensor([5,5,5]))      
print(t)

#Matrix multiplication
x1 = torch.rand((2,3))
x2 = torch.rand((3,4))
print(x1.mm(x2))
print(torch.mm(x1, x2))

#matrix exponentiation
print(x1.matrix_power(3))

# element wise mul.
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# Batch Matrix Multiplication
batch = 32
m = 10
n = 20
p = 30

tensor1 = torch.rand((batch, m, n)) 
tensor2 = torch.rand((batch, n, p))
out_bmm = torch.bmm(tensor1, tensor2)      # (batch, n, p)

# Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
z = x1 - x2

# Other Operations
z = torch.clamp(x, min = 0, max = 10)    # clipped
print(z)

# Useful Operations
x = torch.arange(10)
print(torch.where(x > 5, x, x*2))
print(torch.tensor([0,0,1,1,2,2]).unique())

# Reshaping
x = torch.arange(12)
x_3x3 = x.view(3,4)
x_3x3 = x.reshape(3,4)
print(x_3x3)

x_3x3.view(-1) #Flatten
x_3x3.ravel()  #Flatten

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)   # shape = (batch, 10) 

z = x.permute(0, 2, 1)   # shape = (batch, 5, 2)