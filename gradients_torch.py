import torch
import torch.nn as nn

# f = w * x

# f = 2 * x

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)



#model prediction




#Manual forward method
#w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# def forward(x):
#    return w * x

#loss

#def loss(y, y_pred):
#    return ((y_pred-y)**2).mean()

#gradient 
#MSE = 1/N * (w*x - y)**2
#dJ/dW = 1/N 2x (w*x)

#def gradient(x,y,y_pred):
#    return torch.dot(2*x, y_pred-y).mean()

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')



#Training
lr=0.01

#Iterations
n_iters=100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_iters):
    y_pred = model(X)

    l = loss(Y, y_pred)

    #gradients = backwards pass
    l.backward() #dl/dW
    #dw = gradient(X,Y,y_pred)

    #updating the weights
    optimizer.step()

    #zero the gradients
    optimizer.zero_grad()
 
    if epoch % 2 == 0:
        [w, b] = model.parameters()
        print(f'Epoch: {epoch+1}  Weight = {w[0][0].item():.3f}, Loss = {l:.8f}')



print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
