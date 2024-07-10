import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

# 단순한 신경망
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
    
    
net = simpleNet()

print("W : ")
print(net.W)

x = np.array([5, 1])
p = net.predict(x)
print("predict : ")
print(p)

t = np.array([0, 1, 0])

print(net.loss(x, t))

# 기울기
def f(W):
    return net.loss(x, t)

print("\ndW : ")
dW = numerical_gradient(f, net.W)
print(dW)