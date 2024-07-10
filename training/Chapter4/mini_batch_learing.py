import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# (60000, 784) 훈련데이터 60,000개, 입력데이터는 784열
print(x_train.shape)
# (60000, 10) 정답 레이블은 10줄짜리 데이터
print(t_train.shape)

train_size  = x_train.shape[0] # 60000
batch_size = 10
# 0이상 60000 미만의 수 중에서 무작위로 10개를 골라냄 
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

