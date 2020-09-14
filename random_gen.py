import random as r

class generator():

    def __init__(self):
        self.N = 1000

    def rand_generator(self) -> list:
        rand_list = []
        for i in range(self.N):
            rand_list.append([r.uniform(-1,1), r.uniform(-1,1)])
        return rand_list