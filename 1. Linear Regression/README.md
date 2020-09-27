# Linear Regression

## Py torch - Simple Linear Regression
Import torch and Linear class
```py
import torch
from torch.nn import Linear
import matplotlib.pyplot as plt
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
Print and change our weight and bias to fit our model
```py
print(model.bias, model.weight)
```
Final code
```py
import torch
from torch.nn import Linear

torch.manual_seed(1)

model = Linear(in_features = 1, out_features = 1)

x = torch.tensor([[2.0], [4.0]]) # multiple input
print('multiple: ', model(x))

print(model.bias, model.weight)
```
## Py torch - Linear Regression with custom dataset
Create dataset
```py
X = torch.randn(100, 1) # 100 rows and 1 column
```
## Scikit
