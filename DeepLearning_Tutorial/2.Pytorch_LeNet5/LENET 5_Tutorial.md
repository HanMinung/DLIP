# Case study of LENET 5 

Tutorial environment : Google colab

Date : 2023.05.07

Purpose : Case study of Convolutional Neural Network : LENET5

-------------------

## 1. Architecture

```python
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv_layers = nn.Sequential(            
            # C1
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            # S2
            nn.MaxPool2d(2, 2),
            # C3
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            # S4
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            # F5
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            # F6
            nn.Linear(120, 84),
            nn.ReLU(),
            # OUTPUT
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # Converting multidimensional data to one dimension for FC operation
        x = self.flatten(x)
        logit = self.fc_layers(x)        
        return logit

model = LeNet5().to(device)
print(model)
```

### 1.1. Overall flow of LeNet5 CNN

* Input image : 32 x 32 x 3
* Input parameter of **Conv2D(num1,num2, num3)** function
  * num1 : # of channel of input 
  * num2 : # of channel of output
  * num3 : Kernel size of convolution
* **Convolution 1**
  * Applying **6** 5 x 5 size filters to image without any padding
  * Applying `Relu` activation function
  * output size : 28 x 28 x 6
* **Maxpool**
  * 2 size pooling with stride 2
  * output size : 14 x 14 x 6
* **Convlution 2** : `nn.Conv2d(6, 16, 5)` 
  * output size : 10 x 10 x 16
  * Applying `Relu` activation function
* **Maxpool** : `nn.MaxPool2d(2, 2)`
  * output size : 5 x 5 x 16
* Convolution - Maxpool - Convolution - Maxpool : LeNet5 includes 4 layers
* **Flatten**
  * Transform convolution layer into 1D vector form
  * Through this process, multi dimension feature map can be transferred as input to fully connected layer
* **fc_layers**
  * Defined fully-connected layers
  * F5 : Connected from 400 input neurons to 120 output neurons (`batch size : 120`)
  * F6 : Connected from 120 input neurons to 84 output neurons (`batch size : 84`)
  * Output layer : Connected from 84 input neurons to 10 output neurons (`batch size : 10`)



### 1.2. Result

<img src="https://user-images.githubusercontent.com/99113269/236662175-d86129e8-6e2b-47ed-96b3-58dbe9c3da74.png" alt="image" style="zoom: 80%;" />

<img src="https://user-images.githubusercontent.com/99113269/236662236-a61d5d90-0bef-4960-a7c8-883db2b77e22.png" alt="image" style="zoom:80%;" />



### 1.3. Loss calculation

* Linear regression : Usually use mean squared error
* Classification : Usually use cross entropy

```python
loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

* Used optimizer : Adam (Adaptive Moment Estimation)
* lr : learning rate



### 1.4. Train function definition

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

* model.train : set model to training mode. Dropout & Batch normalization process runs.
* optimizer.zero_grad : initialize gradient of optimizer
* optimizer.step : update weight of model



### 1.5. Test function definition

```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y_pred=pred.argmax(1);
            test_loss += loss_fn(pred, y).item()
            correct += (y_pred == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

