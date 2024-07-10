import numpy as np

# 오차 제곱 합
def sum_of_squares_for_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 교차 엔트로피 오차
def cross_entropy_error(y, t):
    # delta : log 값이 0이 되는 것을 방지
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))



# test!
y = np.array([0.05, 0.92, 0.12, 0.3])
t = np.array([0, 1, 0, 0])

print(sum_of_squares_for_error(y, t))
print(cross_entropy_error(y, t))