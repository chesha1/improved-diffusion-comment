import torch
import numpy as np
import matplotlib.pyplot as plt
# a = np.random.randint(0, 256, size=(256, 256, 1), dtype=np.uint8)
# plt.imshow(a)
# plt.show()

data = np.load('/tmp/openai-2023-07-07-11-58-18-360752/samples_5x64x64x3.npz')
b= data['arr_0']
c = b[2]
plt.imshow(c)
plt.show()
print("aaa")