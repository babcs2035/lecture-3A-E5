# plot dots on a graph (input: 0.txt, 01.txt, 012.txt)
# output: graph.png
# Usage: python graph.py

import matplotlib.pyplot as plt
import numpy as np

arr1 = []
with open("initial.txt", "r") as f:
    for line in f:
        arr1.append(float(line))

# arr2 = []
# with open("01.txt", "r") as f:
#     for line in f:
#         arr2.append(float(line))

arr3 = []
with open("best.txt", "r") as f:
    for line in f:
        arr3.append(float(line))

plt.plot(arr1, "ro", label="initial")
# plt.plot(arr2, "bo", label="01")
plt.plot(arr3, "go", label="best")
plt.xlim([0, 32])
plt.xlabel("tests")
plt.ylabel("average delay (s)")
plt.legend()
plt.savefig("graph.png")
