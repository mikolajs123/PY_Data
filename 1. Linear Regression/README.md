# Linear Regression

## Py torch - Simple Linear Regression
Import torch and Linear class
```py
import torch
from torch.nn import Linear
import matplotlib.pyplot as plt
import numpy as np
```
Use random seed for **weight** and **bias**
```py
torch.manual_seed(1)
```
Use Linear class and set in_features and out_features responsible for how many inputs produce how many outputs 
```py
model = Linear(in_features = 1, out_features = 1)
```
Make predictions with random **weight** and **bias** using our model
```py
x = torch.tensor([2.0]) # single input
print('single: ', model(x))
x = torch.tensor([[2.0], [4.0]]) # multiple input
print('multiple: ', model(x))
```
Final code
```py
import torch
from torch.nn import Linear

torch.manual_seed(1)

model = Linear(in_features = 1, out_features = 1)

x = torch.tensor([[2.0], [4.0]]) # multiple input
print('multiple: ', model(x))
```
## Py torch - Linear Regression with custom dataset
Create dataset
```py
X = torch.randn(100, 1) * 10 # 100 rows and 1 column and multiply by 10 to increase variation
Y = X + 3 * torch.randn(100, 1)
```
Visualize dataset
```py
plt.plot(X.numpy(), Y.numpy(), 'o') # convert tensor to numpy to visualize
plt.ylabel('y')
plt.xlabel('x')
```
Get bias, weight, range of data and predictions
```py
[w, b] = model.parameters()
w1 = w[0][0].item() # get weight at row 0 and column 0 and item to get not as tensor but as python number
b1 = b[0].item() # get bias at column 0 and item to get not as tensor but as python number
x1 = np.array([-30, 30]) # range of data from plot
y1 = b1 + w1 * x1 # make our predictionts 
```
Visualize data and predictions
```py
plt.plot(x1, y1, 'r')
plt.scatter(X, Y)
plt.show()
```
Final code
```py
import torch
from torch.nn import Linear
import matplotlib.pyplot as plt
import numpy as np

# Create dataset
X = torch.randn(100, 1) * 10 # 100 rows and 1 column and multiply by 10 to increase variation
Y = X + 3 * torch.randn(100, 1)

# Visualize dataset
plt.plot(X.numpy(), Y.numpy(), 'o') # convert tensor to numpy to visualize
plt.ylabel('y')
plt.xlabel('x')

# Get bias, weight, range of data and predictions
[w, b] = model.parameters()
w1 = w[0][0].item() # get weight at row 0 and column 0 and item to get not as tensor but as python number
b1 = b[0].item() # get bias at column 0 and item to get not as tensor but as python number
x1 = np.array([-30, 30]) # range of data from plot
y1 = b1 + w1 * x1 # make our predictionts

# Visualize data and predictions
plt.plot(x1, y1, 'r')
plt.scatter(X, Y)
plt.show()
```
## Scikit
