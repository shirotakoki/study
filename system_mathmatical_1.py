import math
import random as r
import matplotlib.pyplot as plt
from random_gen import generator

class neuro(generator):
    def __init__(self):
        super().__init__()
        self.x_0 : float
        self.x_1 : float
        self.w_0 : float
        self.w_1 : float
        self.w_2 : float

        self.thres : float = 0.0

    def step(self) ->float:
        res: float
        num: float = self.w_0 + (self.x_0 * self.w_1)+ (self.x_1 * self.w_2)
        if (num >= 0.0): res = 1.0
        else: res = 0.0
        return res

    def sigmoid(self) ->float:
        res: float
        num: float = self.w_0 + (self.x_0 * self.w_1)+ (self.x_1 * self.w_2)
        res = 1/(1+math.e**-num)
        return res

    def init_weight(self) -> None:
        self.w_0 = r.random()
        self.w_1 = r.random()
        self.w_2 = r.random()

    def set_data(self, d_0: float, d_1: float) -> None:
        self.x_0 = d_0
        self.x_1 = d_1

    def get_step_color(self, x : float) -> str:
        color : str = "r"
        if x == 0 : color = "g"
        return color

if __name__ == "__main__":
    n = neuro()
    data = n.rand_generator()
    n.init_weight()
    for i in data:
        n.set_data(i[0], i[1])
        color = n.get_step_color(n.step())
        plt.scatter(i[0], i[1], c=color)

    plt.show()

