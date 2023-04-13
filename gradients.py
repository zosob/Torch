import numpy as np

# f = w * x

# f = 2 * x

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0


#model prediction

def forward(x):
    return w * x

#loss

def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

#gradient
#MSE = 1/N * (w*x - y)**2
#dJ/dW = 1/N 2x (w*x)

def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')



#Training
lr=0.01

#Iterations
n_iters=14

for epoch in range(n_iters):
    y_pred = forward(X)

    l = loss(Y, y_pred)

    #gradients
    dw = gradient(X,Y,y_pred)

    #updating the weights
    w -= lr * dw

    if epoch % 2 == 0:
        print(f'Epoch: {epoch+1}  Weight = {w:.3f}, Loss = {l:.8f}')



print(f'Prediction after training: f(5) = {forward(5):.3f}')
