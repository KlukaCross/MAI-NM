import numpy as np
from matplotlib import pyplot as plt


def polynom_to_string(p):
    res = []
    for i in range(len(p)):
        if p[i] == 0:
            continue
        if res:
            res.append(" + " if p[i] > 0 else " - ")
        res.append(str(abs(p[i])))
        if i > 0:
            res.append("*x")
        if i > 1:
            res.append(f"^{i}")
    return "".join(res)


def draw_polynom(x, y, p):
    fig, ax = plt.subplots(1, 1)
    plot_x = np.linspace(x[0], x[-1], 100)
    plot_y = sum([p[i]*plot_x**i for i in range(len(p))])
    ax.plot(plot_x, plot_y, "-r")
    ax.scatter(x, y)
    plt.show()
