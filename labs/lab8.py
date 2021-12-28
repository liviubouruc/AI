import numpy as np
import matplotlib.pyplot as plt
images = np.zeros((9, 400, 600))
for i in range(9):
    img = np.load('images/car_%d.npy' % i)
    images[i] = img

for i in range(9):
    plt.imshow(images[i], cmap='gray')
    plt.show()