# Linear Regression

## Py torch - Simple Linear Regression
import torch and Linear class
```py
import torch
from torch.nn import Linear
```
use random seed for **weight** and **bias**
```py
torch.manual_seed(1)
```
use Linear class and set in_features and out_features responsible for how many inputs produce how many outputs 
```py
model = Linear(in_features = 1, out_features = 1)
print(model.bias, model.weight)
```
make predictions with random **weight** and **bias** using our model
```py
x = torch([2.0]) # single input
print('single: ', model(x))
x = torch([[2.0], [4.0]]) # multiple input
print('single: ', model(x))
```
## Scikit
