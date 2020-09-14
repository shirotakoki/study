import numpy as np

classes = tuple(np.arange(10, dtype=np.uint8))
print(classes)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
print(classes)

#上記二つは結果が同じ

test_list = [[] for i in range(10)]

test_list[0].append(1)
test_list[0].append(2)
print(test_list)
