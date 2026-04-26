# generate_circles.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

y = np.where(y == 0, -1, 1)

indices = np.random.permutation(len(y))
X = X[indices]
y = y[indices]

X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

def save_libsvm(X, y, filename):
    with open(filename, 'w') as f:
        for i in range(len(y)):
            line = f"{y[i]} 1:{X[i,0]} 2:{X[i,1]}\n"
            f.write(line)
    print(f"Saved {filename}")

save_libsvm(X_train, y_train, "circles_train.libsvm")
save_libsvm(X_test, y_test, "circles_test.libsvm")

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', label='Positive (+1)', s=20)
plt.scatter(X_train[y_train==-1, 0], X_train[y_train==-1, 1], c='blue', label='Negative (-1)', s=20)
plt.title(f'Training Set (400 samples)')
plt.legend()
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', label='Positive (+1)', s=20)
plt.scatter(X_test[y_test==-1, 0], X_test[y_test==-1, 1], c='blue', label='Negative (-1)', s=20)
plt.title(f'Test Set (100 samples)')
plt.legend()
plt.axis('equal')

plt.tight_layout()
plt.savefig('circles_dataset.png', dpi=150)
print("Saved visualization to circles_dataset.png")
plt.show()