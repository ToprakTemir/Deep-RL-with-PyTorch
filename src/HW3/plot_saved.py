
import matplotlib.pyplot as plt
import numpy as np
import os

draw_type = {"latest": 0, "all": 1}
mode = draw_type["latest"]

all_lists = os.listdir(".")
all_lists = [l for l in all_lists if l.endswith(".pth")]
all_lists.sort()

if len(all_lists) == 0:
    print("No saved data found!")
    exit()

def plot(file_name):
    data = []
    for line in open(file_name, "r"):
        data.append(line.strip().split())
    data = np.array(data).astype(np.float32)

    # average the 6 worker's
    for i in range(0, len(data) - 6 - (len(data) % 6), 6):
        data[i] = np.mean(data[i:i+6], axis=0)
    data = data[::6]


    plt.figure()
    plt.plot(data, label="Training Loss")
    plt.title(file_name)
    plt.xlabel("Sample")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if mode == 0:
    file_name = all_lists[-1]
    plot(file_name)

elif mode == 1:
    for file_name in all_lists:
        plot(file_name)