import math
import random as r
import matplotlib.pyplot as plt
import numpy as np

hnum = 10

def tanh(num : float) ->float:
    res: float
    res = (1-math.e**(-2*num))/(1+math.e**(-2*num))
    return res

def calc_sin(num: float) -> float:
    res: float
    res = 0.8 * math.sin(num)
    return res

def forward_out(x, w, b) -> float:
    res: float
    num = 0
    for i in range(hnum):
        num += x[i]*w[i]
    res = tanh(num + b)
    return res

def forward_hid(i_1, w_1, b) -> float:
    res: float
    res = tanh(i_1 * w_1 + b) 
    return res

def calcdelta_out(out, ans) -> float:
    res: float
    res = -(out - ans) * (1 - out) * (1 + out)
    return res

def calcdelta_hid(out, delta, w) -> float:
    res: float
    res = (1 - out) * (1 + out) * delta * w
    return res

def calc_error(out, ans) -> float:
    res: float
    res = math.pow(out - ans, 2) / 2
    return res

x_1: float = 0.0
t: float = 0.0
eta: float = 0.1

o_1: float = 0.0
h: list = [r.uniform(-1, 1)] * hnum
w_1: list = [r.uniform(-1, 1)] * hnum
b_1: list = [r.uniform(-1, 1)] * hnum
w_2: list = [r.uniform(-1, 1)] * hnum
b_2: float = r.uniform(-1, 1)

epoch = 50
for epoch in range(epoch):
    error = 0
    for i in range(10):
        x_1 = r.uniform(-1, 1)
        t = calc_sin(x_1)

        for j in range(hnum):
            h[j] = forward_hid(x_1, w_1[j], b_1[j])
        
        o_1 = forward_out(h, w_2, b_2)

        del_out = calcdelta_out(o_1, t)
        for j in range(hnum):
            w_2[j] += eta * del_out * h[j]
        b_2 += eta * del_out

        del_hid = [0.0] * hnum
        for j in range(hnum):
            del_hid[j] = calcdelta_hid(h[j], del_out, w_2[j])
            w_1[j] += eta * del_hid[j] * x_1
            b_1[j] += eta * del_hid[j]

        error += calc_error(o_1, t)

    if epoch % 10 == 0:
        print("epoch:",epoch,"error",error/10)
    
for i in range(10):
        x_1 = r.uniform(-1, 1)
        t = calc_sin(x_1)

        for j in range(hnum):
            h[j] = forward_hid(x_1, w_1[j], b_1[j])
        
        o_1 = forward_out(h, w_2, b_2)

        print("input:", x_1, "out:", o_1, "ans:", t)

fig, ax = plt.subplots()
inputs = np.arange(-1, 1, 0.05)
ans_data = np.empty(len(inputs))
out_data = np.empty(len(inputs))

for i in range(len(inputs)):
    ans_data[i] = calc_sin(inputs[i])
    for j in range(hnum):
        h[j] = forward_hid(inputs[i], w_1[j], b_1[j])
    out_data[i] = forward_out(h, w_2, b_2)

ax.plot(inputs, ans_data, label='answer')
ax.plot(inputs, out_data, label='outputs of NN')
ax.set_xlabel("inputs")
ax.set_ylabel("outputs")
ax.legend(loc=0)

plt.show()
        
