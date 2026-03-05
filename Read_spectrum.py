import matplotlib.pyplot as plt

def read_data4(fname)
x = []
y = []

with open("fname", "r") as f:
    for line in f:
        a, b = line.split(",")
        x.append(float(a) * 1e-9)
        y.append(float(b))

plt.plot(x, y)
plt.show()