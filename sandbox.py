import numpy as np

classes = tuple(np.arange(10, dtype=np.uint8))
print(classes)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
print(classes)

#上記二つは結果が同じ
