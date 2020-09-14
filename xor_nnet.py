import math
import random as r
import matplotlib.pyplot as plt

def sigmoid(num : float) ->float:
    res: float
    res = 1/(1+math.e**-num)
    return res

def forward(i_1, i_2, w_1, w_2,b) -> float:
    res: float
    res = sigmoid(i_1 * w_1 + i_2 * w_2 +b)
    return res

def calcdelta_out(out, ans) -> float:
    res: float
    epsilon: float = 1.0
    res = -epsilon * (out - ans) * (1 - out) * out
    return res

def calcdelta_hid(out, delta, w) -> float:
    res: float
    epsilon: float = 1.0
    res = epsilon * (1 - out) * out * delta * w
    return res

def calc_error(out, ans) -> float:
    res: float
    res = math.pow(out - ans, 2) / 2
    return res

x_1: list = [1, 1, 0, 0]
x_2: list = [1, 0, 1, 0]
t: list = [0, 1, 1, 0]
eta: float = 0.1

o_1: float = 0.0
h_1: float = 0.0
h_2: float = 0.0

w_1_11: float = r.uniform(-1,1)
w_1_21: float = r.uniform(-1,1)
w_1_12: float = r.uniform(-1,1)
w_1_22: float = r.uniform(-1,1)
b_1_1: float = r.uniform(-1, 1)
b_1_2: float = r.uniform(-1,1)

w_2_11: float = r.uniform(-1,1)
w_2_12: float = r.uniform(-1, 1)
b_2_1: float = r.uniform(-1, 1)

index = list(range(4))
epoch = 10000
for epoch in range(epoch):
    r.shuffle(index)
    error = 0
    for i in index:
        h_1 = forward(x_1[i], x_2[i], w_1_11, w_1_12, b_1_1)
        h_2 = forward(x_1[i], x_2[i], w_1_21, w_1_22, b_1_2)
        o_1 = forward(h_1, h_2, w_2_11, w_2_12, b_2_1)

        del_out = calcdelta_out(o_1, t[i])
        w_2_11 = w_2_11 + (eta * del_out * h_1)
        w_2_12 = w_2_12 + (eta * del_out * h_2)
        b_2_1 = b_2_1 + (eta * del_out)

        del_1 = calcdelta_hid(h_1, del_out, w_2_11)
        del_2 = calcdelta_hid(h_2, del_out, w_2_12)

        w_1_11 = w_1_11 + (eta * del_1 * x_1[i])
        w_1_21 = w_1_21 + (eta * del_2 * x_1[i])
        w_1_12 = w_1_12 + (eta * del_1 * x_2[i])
        w_1_22 = w_1_22 + (eta * del_2 * x_2[i])
        b_1_1 = b_1_1 + (eta * del_1)
        b_1_2 = b_1_2 + (eta * del_2)

        error += calc_error(o_1, t[i])

    if epoch % 10000 == 0:
        print("epoch:",epoch,"error",error/4)
    

for i in range(4):
        h_1 = forward(x_1[i], x_2[i], w_1_11, w_1_12, b_1_1)
        h_2 = forward(x_1[i], x_2[i], w_1_21, w_1_22, b_1_2)
        o_1 = forward(h_1, h_2, w_2_11, w_2_12, b_2_1)

        print("input:", x_1[i], x_2[i], "hidden:", h_1, h_2, "out:", o_1)
        
