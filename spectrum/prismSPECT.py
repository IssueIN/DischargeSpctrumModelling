import matplotlib.pyplot as plt
import numpy as np
import os

#change to your own dir
folder_path = os.path.join('data', 'prismSPECT', 'test')
file_path = os.path.join(folder_path, 'lineprof_0013_0011.dat')

data = []
with open(file_path, 'r') as file:
    for line in file:
        if line.startswith("#") or line.strip() =="":
            continue
        data.append([float(value) for value in line.split()])
    
data = np.array(data)

x = data[:, 0]
y_columns = data[:, 1:]

for i in range(y_columns.shape[1]):
    plt.plot(x, y_columns[:, i], label=f"Column {i+2}")
    plt.xlabel("Column 1 (x-values)")
    plt.ylabel(f"Column {i+2} Values")
    plt.title(f"Column {i+2} Values")
    plt.legend()
    plt.grid()

    output_file = os.path.join(folder_path, f"plot_column_{i+2}.png")
    plt.savefig(output_file)
    plt.close()
