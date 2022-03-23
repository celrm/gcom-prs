import numpy as np
import matplotlib.pyplot as plt

muestra = np.array([0, 1, 0, 0, 0, 2, 1, 1, 0, 2, 0, 1, 2, 0, 2, 0, 1, 1])

muestra.sort()
plt.plot(range(len(muestra)),muestra)
plt.show()