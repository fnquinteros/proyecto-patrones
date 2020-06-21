import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from classification import svc, Xt, yt
import matplotlib.pyplot as plt
import numpy as np

y_predicted = svc.predict(Xt)

matrix = confusion_matrix(yt, y_predicted)
print(matrix)

fig, ax = plt.subplots()
im = ax.imshow(matrix)

ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))

ax.set_xticklabels(range(3))
ax.set_yticklabels(range(3))

for i in range(3):
    for j in range(3):
        if i != j:
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="w")
        else:
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="black")

ax.set_title("Confusion Matrix for SVC")
fig.tight_layout()
plt.show()
