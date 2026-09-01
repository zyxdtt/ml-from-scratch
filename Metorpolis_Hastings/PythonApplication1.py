import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("samples.txt")
plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.3)
plt.show()