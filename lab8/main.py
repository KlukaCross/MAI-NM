import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib import cm

a = 1

lx = np.pi / 4
ly = np.log(2)

phi_1 = lambda y, t: np.cosh(1 * y) * np.exp(-3 * a * t)
phi_2 = lambda y, t: 0
phi_3 = lambda x, t: np.cos(2 * x) * np.exp(-3 * a * t)
phi_4 = lambda x, t: 5 / 4 * np.cos(2 * x) * np.exp(-3 * a * t)
psi = lambda x, y: np.cos(2 * x) * np.cosh(1 * y)
analytic_solution = lambda x, y, t: np.cos(2 * x) * np.cosh(y) * np.exp(-3 * a * t)

nx = 25
ny = 25
hx = lx / nx
hy = ly / ny

tau = 0.01


def three_diag(A, b):
    n = len(A)

    v = [0 for _ in range(n)]
    u = [0 for _ in range(n)]
    v[0] = A[0][1] / -A[0][0]
    u[0] = b[0] / A[0][0]
    for i in range(1, n-1):
        v[i] = A[i][i+1] / (-A[i][i] - A[i][i-1] * v[i-1])
        u[i] = (A[i][i-1] * u[i-1] - b[i]) / (-A[i][i] - A[i][i-1] * v[i-1])
    v[n-1] = 0
    u[n-1] = (A[n-1][n-2] * u[n-2] - b[n-1]) / (-A[n-1][n-1] - A[n-1][n-2] * v[n-2])

    x = [0 for _ in range(n)]
    x[n-1] = u[n-1]
    for i in range(n-1, 0, -1):
        x[i-1] = v[i-1] * x[i] + u[i-1]
    return np.array(x)


def alternating_directions_method():
    x = np.arange(0, lx + hx, hx)
    y = np.arange(0, ly + hy, hy)
    t = np.arange(0, np.pi + tau, tau)
    res = np.zeros((len(t), len(x), len(y)))

    # Инициализация начального условия
    for x_id in range(len(x)):
        for y_id in range(len(y)):
            res[0][x_id][y_id] = psi(x[x_id], y[y_id])

    # Решение уравнения методом поочередных направлений
    for t_id in range(1, len(t)):
        U_halftime = np.zeros((len(x), len(y)))

        # Условия на границах по x
        for x_id in range(len(x)):
            res[t_id][x_id][0] = phi_3(x[x_id], t[t_id])
            res[t_id][x_id][-1] = phi_4(x[x_id], t[t_id])
            U_halftime[x_id][0] = phi_3(x[x_id], t[t_id] - tau / 2)
            U_halftime[x_id][-1] = phi_4(x[x_id], t[t_id] - tau / 2)

        # Условия на границах по y
        for y_id in range(len(y)):
            res[t_id][0][y_id] = phi_1(y[y_id], t[t_id])
            res[t_id][-1][y_id] = phi_2(y[y_id], t[t_id])
            U_halftime[0][y_id] = phi_1(y[y_id], t[t_id] - tau / 2)
            U_halftime[-1][y_id] = phi_2(y[y_id], t[t_id] - tau / 2)

        # внутренних узлов сетки по y
        for y_id in range(1, len(y) - 1):
            A = np.zeros((len(x) - 2, len(x) - 2))
            b = np.zeros((len(x) - 2))

            # Формирование матрицы коэффициентов для метода прогонки
            A[0][0] = 2 * hx ** 2 * hy ** 2 + 2 * a * tau * hy ** 2
            A[0][1] = -a * tau * hy ** 2
            for i in range(1, len(A) - 1):
                A[i][i - 1] = -a * tau * hy ** 2
                A[i][i] = 2 * hx ** 2 * hy ** 2 + 2 * a * tau * hy ** 2
                A[i][i + 1] = -a * tau * hy ** 2
            A[-1][-2] = -a * tau * hy ** 2
            A[-1][-1] = 2 * hx ** 2 * hy ** 2 + 2 * a * tau * hy ** 2

            # Формирование правой части для метода прогонки
            for x_id in range(1, len(x) - 1):
                b[x_id - 1] = (
                        res[t_id - 1][x_id][y_id - 1] * a * tau * hx ** 2
                        + res[t_id - 1][x_id][y_id] * (2 * hx ** 2 * hy ** 2 - 2 * a * tau * hx ** 2)
                        + res[t_id - 1][x_id][y_id + 1] * a * tau * hx ** 2
                )
            b[0] -= (-a * tau * hy ** 2) * phi_1(y[y_id], t[t_id] - tau / 2)
            b[-1] -= (-a * tau * hy ** 2) * phi_2(y[y_id], t[t_id] - tau / 2)
            U_halftime[1:-1, y_id] = np.array(three_diag(A, b))

        # То же самое для x
        for x_id in range(1, len(x) - 1):
            A = np.zeros((len(y) - 2, len(y) - 2))
            b = np.zeros((len(y) - 2))

            A[0][0] = 2 * hx ** 2 * hy ** 2 + 2 * a * tau * hx ** 2
            A[0][1] = -a * tau * hx ** 2
            for i in range(1, len(A) - 1):
                A[i][i - 1] = -a * tau * hx ** 2
                A[i][i] = 2 * hx ** 2 * hy ** 2 + 2 * a * tau * hx ** 2
                A[i][i + 1] = -a * tau * hx ** 2
            A[-1][-2] = -a * tau * hx ** 2
            A[-1][-1] = 2 * hx ** 2 * hy ** 2 + 2 * a * tau * hx ** 2

            for y_id in range(1, len(y) - 1):
                b[y_id - 1] = (
                        U_halftime[x_id - 1][y_id] * a * tau * hy ** 2
                        + U_halftime[x_id][y_id] * (2 * hx ** 2 * hy ** 2 - 2 * a * tau * hy ** 2)
                        + U_halftime[x_id + 1][y_id] * a * tau * hy ** 2
                )
            b[0] -= (-a * tau * hx ** 2) * phi_3(x[x_id], t[t_id])
            b[-1] -= (-a * tau * hx ** 2) * phi_4(x[x_id], t[t_id])
            res[t_id][x_id][1:-1] = three_diag(A, b)

    return res


def fractional_steps_method():
    x = np.arange(0, lx + hx, hx)
    y = np.arange(0, ly + hy, hy)
    t = np.arange(0, np.pi + tau, tau)
    res = np.zeros((len(t), len(x), len(y)))

    # Инициализация начального условия
    for x_id in range(len(x)):
        for y_id in range(len(y)):
            res[0][x_id][y_id] = psi(x[x_id], y[y_id])

    # Решение уравнения методом дробных шагов
    for t_id in range(1, len(t)):
        U_halftime = np.zeros((len(x), len(y)))

        for x_id in range(len(x)):
            res[t_id][x_id][0] = phi_3(x[x_id], t[t_id])
            res[t_id][x_id][-1] = phi_4(x[x_id], t[t_id])
            U_halftime[x_id][0] = phi_3(x[x_id], t[t_id] - tau / 2)
            U_halftime[x_id][-1] = phi_4(x[x_id], t[t_id] - tau / 2)

        for y_id in range(len(y)):
            res[t_id][0][y_id] = phi_1(y[y_id], t[t_id])
            res[t_id][-1][y_id] = phi_2(y[y_id], t[t_id])
            U_halftime[0][y_id] = phi_1(y[y_id], t[t_id] - tau / 2)
            U_halftime[-1][y_id] = phi_2(y[y_id], t[t_id] - tau / 2)

        for y_id in range(1, len(y) - 1):
            A = np.zeros((len(x) - 2, len(x) - 2))
            b = np.zeros((len(x) - 2))

            A[0][0] = hx ** 2 + 2 * a * tau
            A[0][1] = -a * tau
            for i in range(1, len(A) - 1):
                A[i][i - 1] = -a * tau
                A[i][i] = hx ** 2 + 2 * a * tau
                A[i][i + 1] = -a * tau
            A[-1][-2] = -a * tau
            A[-1][-1] = hx ** 2 + 2 * a * tau

            for x_id in range(1, len(x) - 1):
                b[x_id - 1] = res[t_id - 1][x_id][y_id] * hx ** 2
            b[0] -= (-a * tau) * phi_1(y[y_id], t[t_id] - tau / 2)
            b[-1] -= (-a * tau) * phi_2(y[y_id], t[t_id] - tau / 2)
            U_halftime[1:-1, y_id] = np.array(three_diag(A, b))

        for x_id in range(1, len(x) - 1):
            A = np.zeros((len(y) - 2, len(y) - 2))
            b = np.zeros((len(y) - 2))

            A[0][0] = hy ** 2 + 2 * a * tau
            A[0][1] = -a * tau
            for i in range(1, len(A) - 1):
                A[i][i - 1] = -a * tau
                A[i][i] = hy ** 2 + 2 * a * tau
                A[i][i + 1] = -a * tau
            A[-1][-2] = -a * tau
            A[-1][-1] = hy ** 2 + 2 * a * tau

            for y_id in range(1, len(y) - 1):
                b[y_id - 1] = U_halftime[x_id][y_id] * hy ** 2
            b[0] -= (-a * tau) * phi_3(x[x_id], t[t_id])
            b[-1] -= (-a * tau) * phi_4(x[x_id], t[t_id])
            res[t_id][x_id][1:-1] = three_diag(A, b)

    return res


def AnalyticSolution():
    x = np.arange(0, lx + hx, hx)
    y = np.arange(0, ly + hy, hy)
    t = np.arange(0, np.pi + tau, tau)
    return np.array([np.array([np.array([analytic_solution(xi, yi, ti) for yi in y]) for xi in x]) for ti in t])



ANALYTIC_SOLUTION_METHOD_NAME = 'Analytic'
answers = {
    ANALYTIC_SOLUTION_METHOD_NAME: AnalyticSolution(),
    'Alternating Directions': alternating_directions_method(),
    'Fractional Steps': fractional_steps_method(),
}

def mae(numeric):
    analytic_solution = answers[ANALYTIC_SOLUTION_METHOD_NAME]
    return np.array([np.abs(numeric_el - analytic_el).mean() for numeric_el, analytic_el in zip(numeric, analytic_solution)]).max()


x = np.arange(0, lx + hx, hx)
y = np.arange(0, ly + hy, hy)

def plot_results_x(ax, cur_x, timestamp=0):
    cur_x_id = abs(x - cur_x).argmin()

    for method_name, answer in answers.items():
        solution = answer
        ax.plot(y, solution[timestamp][cur_x_id], label=method_name)

    ax.set_xlabel('y')
    ax.set_ylabel('U')
    ax.grid()
    ax.legend(loc='best')

def plot_results_y(ax, cur_y, timestamp=0):
    cur_y_id = abs(y - cur_y).argmin()

    for method_name, answer in answers.items():
        solution = answer
        ax.plot(x, [solution[timestamp][i][cur_y_id] for i in range(len(x))], label=method_name)

    ax.set_xlabel('x')
    ax.set_ylabel('U')
    ax.grid()
    ax.legend(loc='best')

def plot_errors_from_y(ax, timestamp=0):
    analytic_solution = answers[ANALYTIC_SOLUTION_METHOD_NAME]
    analytic_xn = np.array([[analytic_solution[timestamp][j][i] for j in range(len(x))] for i in range(len(y))])

    for method_name, answer in answers.items():
        if method_name == ANALYTIC_SOLUTION_METHOD_NAME:
            continue

        solution = answer
        solution_xn = np.array([[solution[timestamp][j][i] for j in range(len(x))] for i in range(len(y))])
        max_abs_errors = np.array([
            np.abs(solution_i - analytic_i).max()
            for solution_i, analytic_i in zip(solution_xn,analytic_xn)
        ])
        ax.plot(y, max_abs_errors, label=method_name)

    ax.grid()
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('Max abs error')

def plot_errors_from_x(ax, timestamp=0):
    analytic_solution = answers[ANALYTIC_SOLUTION_METHOD_NAME]
    analytic_yn = np.array([analytic_solution[timestamp][i] for i in range(len(x))])

    for method_name, answer in answers.items():
        if method_name == ANALYTIC_SOLUTION_METHOD_NAME:
            continue

        solution = answer
        solution_yn = np.array([solution[timestamp][i] for i in range(len(x))])
        max_abs_errors = np.array([
            np.abs(solution_i - analytic_i).max()
            for solution_i, analytic_i in zip(solution_yn,analytic_yn)
        ])
        ax.plot(x, max_abs_errors, label=method_name)

    ax.grid()
    ax.legend(loc='best')
    ax.set_xlabel('y')
    ax.set_ylabel('Max abs error')

def calculate_mae(answer, analytic_solution):
    x = np.arange(0, lx + hx, hx)
    y = np.arange(0, ly + hy, hy)
    t = np.arange(0, np.pi + tau, tau)

    errors = []
    for ti in range(len(t)):
        mx = -1
        for xi in range(len(x)):
            for yi in range(len(y)):
                mx = max(np.abs(answer[ti][xi][yi] - analytic_solution[ti][xi][yi]).mean(), mx)
        errors.append(mx)
    return errors

def plot_errors_from_t(ax):
    analytic_solution = answers[ANALYTIC_SOLUTION_METHOD_NAME]

    t = np.arange(0, np.pi + tau, tau)

    for method_name, answer in answers.items():
        if method_name == ANALYTIC_SOLUTION_METHOD_NAME:
            continue
        errors = calculate_mae(answer, analytic_solution)
        ax.plot(t, errors, label=method_name)

    ax.grid()
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.set_ylabel('Max abs error')


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
timestamp = 5
plot_results_x(ax[0][0], cur_x=lx/2, timestamp=timestamp)
plot_results_y(ax[0][1], cur_y=ly/2, timestamp=timestamp)
plot_errors_from_t(ax[1][0])
plot_errors_from_x(ax[1][1], timestamp=timestamp)

plt.tight_layout()
plt.show()