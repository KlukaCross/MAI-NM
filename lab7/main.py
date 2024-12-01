from enum import StrEnum
from math import *

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class SolutionMethod(StrEnum):
    seidel = "Метод Зейделя"
    simple = "Метод простых итераций"
    relax = "Метод верхней релаксации"
    exact = "Аналитическое решение"


def interpolate_linearly(start_val, end_val, num_points):
    """Функция линейной интерполяции между двумя точками."""
    return np.linspace(start_val, end_val, num_points)


def interpolate_2d_matrix(res):
    n, m = res.shape

    # Интерполяция для верхней и левой границ, если они заданы
    if res[0, 1:-1].any() and res[1:-1, 0].any():  # res[0][j] и res[i][0]
        for i in range(1, n):
            for j in range(1, m):
                res[i, j] = interpolate_linearly(res[0, j], res[i, 0], i + 1 + j + 1 - 1)[i]

    # Интерполяция для верхней и правой границ, если они заданы
    elif res[0, 1:-1].any() and res[1:-1, -1].any():  # res[0][j] и res[i][-1]
        for i in range(1, n):
            for j in range(0, m - 1):
                res[i, j] = interpolate_linearly(res[0, j], res[i, -1], i + 1 + (m - j) - 1)[i]

    # Интерполяция для нижней и правой границ, если они заданы
    elif res[-1, 1:-1].any() and res[1:-1, -1].any():  # res[-1][j] и res[i][-1]
        for i in range(0, n - 1):
            for j in range(0, m - 1):
                res[i, j] = interpolate_linearly(res[-1, j], res[i, -1], (n - i) + (m - j) - 1)[n - i - 1]

    # Интерполяция для нижней и левой границ, если они заданы
    elif res[-1, 1:-1].any() and res[1:-1, 0].any():  # res[-1][j] и res[i][0]
        for i in range(0, n - 1):
            for j in range(1, m):
                res[i, j] = interpolate_linearly(res[-1, j], res[i, 0], (n - i) + j + 1 - 1)[n - i - 1]

    # Интерполяция для верхней и нижней границ, если хотя бы одна из них задана
    elif res[0, 1:-1].any() or res[-1, 1:-1].any():  # res[0][j] и res[-1][j]
        for j in range(m):
            res[1:-1, j] = interpolate_linearly(res[0, j], res[-1, j], n)[1:-1]

    # Интерполяция для левой и правой границ, если хотя бы одна из них задана
    elif res[1:-1, 0].any() or res[1:-1, -1].any():  # res[i][0] и res[i][-1]
        for i in range(n):
            res[i, 1:-1] = interpolate_linearly(res[i, 0], res[i, -1], m)[1:-1]

    else:
        print(
            "От interpolate_2d_matrix: ваша матрицу нельзя проинтерполировать ни одним из имеющихся вариантов интерполяции, так что матрица осталась той же"
        )

    return res


def sign(x):
    return int(x > 0) - int(x < 0)


def finite_difference_schema_general_view(
    x_range,  # (x_begin, x_end) диапазон значений x
    y_range,  # (y_begin, y_end) диапазон значений y
    h_x,  # Длина шага по x
    h_y,  # Длина шага по y
    a,  # Коэффициент перед ∂u/∂x
    b,  # Коэффициент перед ∂u/∂y
    c,  # Коэффициент перед u
    f,  # Источниковый член в выражении ∂²u/∂x² + ∂²u/∂y² + a * ∂u/∂x + b * ∂u/∂y = f
    alpha_1,  # Коэффициент перед ∂/∂x u(0, y)
    beta_1,  # Коэффициент перед u(0, y)
    alpha_2,  # Коэффициент перед ∂/∂x u(l₁, y)
    beta_2,  # Коэффициент перед u(l₁, y)
    alpha_3,  # Коэффициент перед ∂/∂y u(x, 0)
    beta_3,  # Коэффициент перед u(x, 0)
    alpha_4,  # Коэффициент перед ∂/∂y u(x, l₂)
    beta_4,  # Коэффициент перед u(x, l₂)
    phi_1,  # Граничное условие для 0 по x
    phi_2,  # Граничное условие для l₁ по x
    phi_3,  # Граничное условие для 0 по y
    phi_4,  # Граничное условие для l₂ по y
    initial,  # флаг управления линейной интерполяции неизвестных компонент
    method_name,  # название используемого метода
    theta, # коэффициент метода верхней релаксации (для получения просто Зейделя нужен коэффициент = 1), в методе простых итераций не используется
    eps,  # точность решения
    direction_of_traversing_the_matrix,  # ↘↙↖↗ - всевозможные обходы матрицы
):
    """
    Решает эллиптическое дифференциальное уравнения, используя конечно-разностную схему и один из численных методов для системы уравнений.
    Возвращает матрицу U со значениями функции
    """

    x = np.arange(*x_range, h_x)
    y = np.arange(*y_range, h_y)
    n = len(x)
    m = len(y)

    res = np.zeros((n, m))  # Такое объявление означает, что x отвечает за вектор направленный ровно вниз, а y за вектор, направленный вправо
    # Если перефразировать: x - первый индекс, y - второй индекс двумерной матрицы

    # Шаг 1. Инициализация сетки с граничными условиями:

    # Строки 0, m -> используем граничные условия для x (если они первого рода α_i = 0)
    for cur_y_id in range(m):
        if alpha_1 == 0:
            res[0][cur_y_id] = 1 / beta_1 * phi_1(y[cur_y_id])
        if alpha_2 == 0:
            res[-1][cur_y_id] = 1 / beta_2 * phi_2(y[cur_y_id])

    # Столбцы 0, n -> используем граничные условия для y (если они первого рода α_i = 0)
    for cur_x_id in range(n):
        if alpha_3 == 0:
            res[cur_x_id][0] = 1 / beta_3 * phi_3(x[cur_x_id])
        if alpha_4 == 0:
            res[cur_x_id][-1] = 1 / beta_4 * phi_4(x[cur_x_id])

    # Интерполяцию делаем относительно известных строк или столбцов
    if initial == True:
        interpolate_2d_matrix(res)

    iters = 1
    while True:
        res_prev = res
        res = np.zeros((n, m))
        # Шаг 1. Инициализация сетки с граничными условиями первого порядка
        # Строки 0, m -> используем граничные условия для x (если они первого рода α_i = 0)
        for cur_y_id in range(m):
            if alpha_1 == 0:
                res[0][cur_y_id] = 1 / beta_1 * phi_1(y[cur_y_id])
            if alpha_2 == 0:
                res[-1][cur_y_id] = 1 / beta_2 * phi_2(y[cur_y_id])
        # Столбцы 0, n -> используем граничные условия для y (если они первого рода α_i = 0)
        for cur_x_id in range(n):
            if alpha_3 == 0:
                res[cur_x_id][0] = 1 / beta_3 * phi_3(x[cur_x_id])
            if alpha_4 == 0:
                res[cur_x_id][-1] = 1 / beta_4 * phi_4(x[cur_x_id])

        # Шаг 2. Внутренние точки посчитаем через точки предыдущей матрицы
        all_traversings = "↘↙↖↗"
        traversing_information = [  # Описание четвёрок (x_start, x_end, y_start, y_end) у каждого направления обхода
            (1, n - 2, 1, m - 2),  #↘ сверху вниз по [1, n - 2], слева направо по [1, m - 2]
            (1, n - 2, m - 2, 1),  #↙ сверху вниз по [1, n - 2], справа налево по [m - 2, 1] (ОО)
            (n - 2, 1, m - 2, 1),  #↖ снизу вверх по [n - 2, 1] (ОО), снизу вверх по [m - 2, 1] (ОО)
            (n - 2, 1, 1, m - 2),  #↗ снизу вверх по [n - 2, 1] (ОО), слева направо по [1, m - 2]
        ]  # ОО - обратный обход
        index_of_traversing = all_traversings.find(direction_of_traversing_the_matrix)
        x_start, x_end, y_start, y_end = traversing_information[index_of_traversing]
        x_dir = sign(x_end - x_start)  # Узнаём направление обхода по x
        y_dir = sign(y_end - y_start)  # Узнаём направление обхода по y

        uij_coeff = (c - 2 / h_x ** 2 - 2 / h_y ** 2)
        for cur_x_id in range(x_start, x_end + x_dir,
                              x_dir):  # Python не включает граничную току поэтому для её рассмотрения делаем шаг дальше
            for cur_y_id in range(y_start, y_end + y_dir, y_dir):
                if method_name == SolutionMethod.simple:
                    part_d2u_dx2 = 1 / h_x ** 2 * (
                            res_prev[cur_x_id + x_dir][cur_y_id] + res_prev[cur_x_id - x_dir][cur_y_id])
                    part_d2u_dy2 = 1 / h_y ** 2 * (
                            res_prev[cur_x_id][cur_y_id + y_dir] + res_prev[cur_x_id][cur_y_id - y_dir])
                    du_dx = a / (2 * h_x) * (
                            res_prev[cur_x_id + x_dir][cur_y_id] - res_prev[cur_x_id - x_dir][cur_y_id])
                    du_dy = b / (2 * h_y) * (
                            res_prev[cur_x_id][cur_y_id + y_dir] - res_prev[cur_x_id][cur_y_id - y_dir])

                    res[cur_x_id][cur_y_id] = 1 / uij_coeff * (
                            f(x[cur_x_id], y[cur_y_id]) - (part_d2u_dx2 + part_d2u_dy2 + du_dx + du_dy))

                elif method_name in [SolutionMethod.seidel, SolutionMethod.relax]:
                    part_d2u_dx2 = 1 / h_x ** 2 * (
                            res_prev[cur_x_id + x_dir][cur_y_id] + res[cur_x_id - x_dir][cur_y_id])
                    part_d2u_dy2 = 1 / h_y ** 2 * (
                            res_prev[cur_x_id][cur_y_id + y_dir] + res[cur_x_id][cur_y_id - y_dir])
                    du_dx = a / (2 * h_x) * (res_prev[cur_x_id + x_dir][cur_y_id] - res[cur_x_id - x_dir][cur_y_id])
                    du_dy = b / (2 * h_y) * (res_prev[cur_x_id][cur_y_id + y_dir] - res[cur_x_id][cur_y_id - y_dir])

                    res[cur_x_id][cur_y_id] = (
                            theta * (1 / uij_coeff * (
                            f(x[cur_x_id], y[cur_y_id]) - (part_d2u_dx2 + part_d2u_dy2 + du_dx + du_dy))) +
                            (1 - theta) * res_prev[cur_x_id][cur_y_id]
                    )

                else:
                    raise ValueError(
                        f"Неправильное название метода, {method_name} не подходит ни под один из списка:\nМетод простых итераций\nМетод Зейделя\nМетод верхней релаксации")

        # Шаг 3. Инициализация сетки с граничными условиями второго и третьего порядков (α_i ≠ 0)
        for cur_y_id in range(1, m - 1):
            if alpha_1 != 0:
                u0j_coef = 2 * h_x * beta_1 - 3 * alpha_1
                res[0][cur_y_id] = 1 / u0j_coef * (
                        2 * h_x * phi_1(y[cur_y_id]) - alpha_1 * (4 * res[1][cur_y_id] - res[2][cur_y_id]))
            if alpha_2 != 0:
                unj_coeff = 2 * h_x * beta_2 + 3 * alpha_2
                res[-1][cur_y_id] = 1 / unj_coeff * (
                        2 * h_x * phi_2(y[cur_y_id]) + alpha_2 * (4 * res[-2][cur_y_id] - res[-3][cur_y_id]))

        for cur_x_id in range(1, n - 1):
            if alpha_3 != 0:
                ui0_coef = 2 * h_y * beta_3 - 3 * alpha_3
                res[cur_x_id][0] = 1 / ui0_coef * (
                        2 * h_y * phi_3(x[cur_x_id]) - alpha_3 * (4 * res[cur_x_id][1] - res[cur_x_id][2]))
            if alpha_4 != 0:
                uim_coeff = 2 * h_y * beta_4 + 3 * alpha_4
                res[cur_x_id][-1] = 1 / uim_coeff * (
                        2 * h_y * phi_4(x[cur_x_id]) + alpha_4 * (4 * res[cur_x_id][-2] - res[cur_x_id][-3]))

        if L2_norm_diff(res, res_prev) < eps:
            break

        iters += 1

    return res, iters


def get_analytical_solution(
        x_range,  # (x_begin, x_end) диапазон значений x
        y_range,  # (y_begin, y_end) диапазон значений y
        h_x,  # Длина шага по x
        h_y,  # Длина шага по y
):
    """
    Получает аналитическое решение элиптического дифференциального уравнения
    Возвращает матрицу U со значениями функции
    """
    x = np.arange(*x_range, h_x)
    y = np.arange(*y_range, h_y)

    res = np.zeros((len(x), len(y)))
    for idx in range(len(x)):
        for idy in range(len(y)):
            res[idx][idy] = solution(x[idx], y[idy])

    return res


def max_abs_error(A, B):
    """
    Вычисляет модуль абсолютной ошибки элементов матрицы A Относительно элементов матрицы B
    """
    assert A.shape == B.shape, "Матрицы должны быть одинакового размера"
    return abs(A - B).max()


def mean_abs_error(A, B):
    """
    Вычисляет модуль средней ошибки элементов матрицы A Относительно элементов матрицы B
    """
    assert A.shape == B.shape, "Матрицы должны быть одинакового размера"
    return abs(A - B).mean()


def L2_norm_diff(A, B):
    """
    Вычисляет L2-норму разницы двух матриц A и B.
    """
    assert A.shape == B.shape, "Матрицы должны быть одинакового размера"
    diff = A - B
    return np.sqrt(np.sum(diff ** 2))


def plot_combined_results_and_errors(
    solutions,  # словарь решений: solutions[имя_метода] = численное решение
    analytical_solution_name,  # имя аналитического решения для сравнения
    x_range,  # диапазон значений x
    y_range,  # диапазон значений y
    h_x,  # длина шага по x
    h_y  # длина шага по y
):
    x = np.arange(*x_range, h_x)
    y = np.arange(*y_range, h_y)

    # Ограничиваем количество возможных значений для улучшения производительности
    max_slices = 100
    if len(x) > max_slices:
        step_x = len(x) // max_slices
        x_reduced = x[::step_x]
    else:
        x_reduced = x

    if len(y) > max_slices:
        step_y = len(y) // max_slices
        y_reduced = y[::step_y]
    else:
        y_reduced = y

    # Создаем fig и разбиение на 2х2 подграфика
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(bottom=0.3)

    # Линии для интерактивных графиков
    lines_x = []
    lines_y = []
    for method_name, solution in solutions.items():
        line_x, = axs[0, 0].plot(y, solution[0, :], label=method_name)
        lines_x.append(line_x)

        line_y, = axs[0, 1].plot(x, solution[:, 0], label=method_name)
        lines_y.append(line_y)

    # Настройка интерактивных графиков
    axs[0, 0].set_title('u(x, y) при изменяющемся x')
    axs[0, 0].set_xlabel('y')
    axs[0, 0].set_ylabel('u(x, y)')
    axs[0, 0].legend()

    axs[0, 1].set_title('u(x, y) при изменяющемся y')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('u(x, y)')
    axs[0, 1].legend()

    # Линии для графиков ошибок
    lines_err_y = axs[1, 0]
    lines_err_x = axs[1, 1]

    for method_name, solution in solutions.items():
        if method_name == analytical_solution_name:
            continue
        # График ошибок по y
        max_abs_errors_y = np.array([
            max_abs_error(solution[:, i], solutions[analytical_solution_name][:, i])
            for i in range(len(y))
        ])
        lines_err_y.plot(y, max_abs_errors_y, label=method_name)

        # График ошибок по x
        max_abs_errors_x = np.array([
            max_abs_error(solution[i, :], solutions[analytical_solution_name][i, :])
            for i in range(len(x))
        ])
        lines_err_x.plot(x, max_abs_errors_x, label=method_name)

    # Настройка графиков ошибок
    lines_err_y.set_title('Max abs error по y')
    lines_err_y.set_xlabel('y')
    lines_err_y.set_ylabel('Max abs error')
    lines_err_y.legend()
    lines_err_y.grid()

    lines_err_x.set_title('Max abs error по x')
    lines_err_x.set_xlabel('x')
    lines_err_x.set_ylabel('Max abs error')
    lines_err_x.legend()
    lines_err_x.grid()

    # Создаем ползунки
    axslider_x = plt.axes([0.15, 0.2, 0.3, 0.03])
    slider_x = Slider(axslider_x, 'x', 0, len(x_reduced) - 1, valinit=0, valfmt='%d')

    axslider_y = plt.axes([0.55, 0.2, 0.3, 0.03])
    slider_y = Slider(axslider_y, 'y', 0, len(y_reduced) - 1, valinit=0, valfmt='%d')

    # Функции обновления интерактивных графиков
    def update_x(val):
        x_idx = int(slider_x.val)
        for line, (method_name, solution) in zip(lines_x, solutions.items()):
            line.set_ydata(solution[x_idx, :])

    fig.canvas.draw_idle()

    def update_y(val):
        y_idx = int(slider_y.val)
        for line, (method_name, solution) in zip(lines_y, solutions.items()):
            line.set_ydata(solution[:, y_idx])
        fig.canvas.draw_idle()

    # Привязываем функции обновления к событиям изменения состояния ползунков
    slider_x.on_changed(update_x)
    slider_y.on_changed(update_y)

    # Показываем график
    plt.show()


# Источниковый член в выражении ∂²u/∂x² + ∂²u/∂y² + a * ∂u/∂x + b * ∂u/∂y + cu = f
def f(x, y):
    return 0


# Граничные условия для x
def phi_1(y):
    return exp(-y) * cos(y)


def phi_2(y):
    return 0


# Граничные условия для y
def phi_3(x):
    return exp(-x) * cos(x)


def phi_4(x):
    return 0


# Аналитическое решение
def solution(x, y):
    return exp(-x - y) * cos(x) * cos(y)


"""
система имеет вид:

∂²u/∂x² + ∂²u/∂y² + a * ∂u/∂x + b * ∂u/∂y + cu = f
α₁ * ∂/∂x u(0, y) + β₁ * u(0, y) = φ₁(y)
α₂ * ∂/∂x u(l₁, y) + β₂ * u(l₁, y) = φ₂(y)
α₃ * ∂/∂y u(x, 0) + β₃ * u(x, 0) = φ₃(x)
α₄ * ∂/∂y u(x, l₂) + β₄ * u(x, l₂) = φ₄(x)
"""


@click.command()
@click.option("--l_1", default=pi / 2, help="граница по x")
@click.option("--l_2", default=pi / 2, help="граница по y")
@click.option("--h_x_desired", default=0.05, help="желаемый шаг по x")
@click.option("--h_y_desired", default=0.05, help="желаемый шаг по y")
@click.option("--init", default=True, help="флаг управления линейной интерполяции неизвестных компонент")
@click.option("--theta", default=1.5,
              help="коэффициент метода верхней релаксации (для получения просто Зейделя нужен коэффициент = 1), в методе простых итераций не используется")
@click.option("--eps", default=1e-3, help="точность решения")
@click.option("--direction_of_traversing_the_matrix", default="↘", help="↘↙↖↗ - всевозможные обходы матрицы")
def main(
    l_1,
    l_2,
    h_x_desired,
    h_y_desired,
    init,
    theta,
    eps,
    direction_of_traversing_the_matrix,
):
    a = 2  # Коэффициент перед ∂u/∂x
    b = 2  # Коэффициент перед ∂u/∂y
    c = 4  # Коэффициент перед u

    alpha_1 = 0  # Коэффициент перед ∂/∂x u(0, y)
    beta_1 = 1  # Коэффициент перед u(0, y)

    alpha_2 = 0  # Коэффициент перед ∂/∂x u(l₁, y)
    beta_2 = 1  # Коэффициент перед u(l₁, y)

    alpha_3 = 0  # Коэффициент перед ∂/∂y u(x, 0)
    beta_3 = 1  # Коэффициент перед u(x, 0)

    alpha_4 = 0  # Коэффициент перед ∂/∂y u(x, l₂)
    beta_4 = 1  # Коэффициент перед u(x, l₂)

    N_x = ceil(l_1 / h_x_desired)
    N_y = ceil(l_2 / h_y_desired)

    h_x = l_1 / (N_x - 1)
    h_y = l_2 / (N_y - 1)

    x_begin = 0
    x_end = l_1 + h_x

    y_begin = 0
    y_end = l_2 + h_y

    analytical_solution = get_analytical_solution(
        x_range=(x_begin, x_end),
        y_range=(y_begin, y_end),
        h_x=h_x,
        h_y=h_y,
    )

    solutions_3 = dict()

    solutions_3[SolutionMethod.exact] = analytical_solution

    print("Итерационный метод:")
    iterative_solution, iterative_iters = finite_difference_schema_general_view(
        x_range=(x_begin, x_end),
        y_range=(y_begin, y_end),
        h_x=h_x,
        h_y=h_y,
        a=a,
        b=b,
        c=c,
        f=f,
        alpha_1=alpha_1,
        beta_1=beta_1,
        alpha_2=alpha_2,
        beta_2=beta_2,
        alpha_3=alpha_3,
        beta_3=beta_3,
        alpha_4=alpha_4,
        beta_4=beta_4,
        phi_1=phi_1,
        phi_2=phi_2,
        phi_3=phi_3,
        phi_4=phi_4,
        method_name=SolutionMethod.simple,
        initial=init,
        theta=0,
        eps=eps,
        direction_of_traversing_the_matrix=direction_of_traversing_the_matrix,
    )

    solutions_3[SolutionMethod.simple] = iterative_solution

    print(f'Модуль максимальной ошибки = {max_abs_error(iterative_solution, analytical_solution)}')
    print(f'Модуль средней ошибки = {mean_abs_error(iterative_solution, analytical_solution)}')
    print(f'Количество итераций = {iterative_iters}')
    print()

    print("Метод Зейделя:")
    seidel_solution, seidel_iters = finite_difference_schema_general_view(
        x_range=(x_begin, x_end),
        y_range=(y_begin, y_end),
        h_x=h_x,
        h_y=h_y,
        a=a,
        b=b,
        c=c,
        f=f,
        alpha_1=alpha_1,
        beta_1=beta_1,
        alpha_2=alpha_2,
        beta_2=beta_2,
        alpha_3=alpha_3,
        beta_3=beta_3,
        alpha_4=alpha_4,
        beta_4=beta_4,
        phi_1=phi_1,
        phi_2=phi_2,
        phi_3=phi_3,
        phi_4=phi_4,
        method_name=SolutionMethod.seidel,
        initial=init,
        theta=1,
        eps=eps,
        direction_of_traversing_the_matrix=direction_of_traversing_the_matrix,
    )

    solutions_3[SolutionMethod.seidel] = seidel_solution

    print(f'Модуль максимальной ошибки = {max_abs_error(seidel_solution, analytical_solution)}')
    print(f'Модуль средней ошибки = {mean_abs_error(seidel_solution, analytical_solution)}')
    print(f'Количество итераций = {seidel_iters}')
    print()

    print("Метод верхней релаксации:")
    relaxation_solution, relaxation_iters = finite_difference_schema_general_view(
        x_range=(x_begin, x_end),
        y_range=(y_begin, y_end),
        h_x=h_x,
        h_y=h_y,
        a=a,
        b=b,
        c=c,
        f=f,
        alpha_1=alpha_1,
        beta_1=beta_1,
        alpha_2=alpha_2,
        beta_2=beta_2,
        alpha_3=alpha_3,
        beta_3=beta_3,
        alpha_4=alpha_4,
        beta_4=beta_4,
        phi_1=phi_1,
        phi_2=phi_2,
        phi_3=phi_3,
        phi_4=phi_4,
        method_name=SolutionMethod.relax,
        initial=init,
        theta=theta,
        eps=eps,
        direction_of_traversing_the_matrix=direction_of_traversing_the_matrix,
    )

    solutions_3[SolutionMethod.relax] = relaxation_solution

    print(f'Модуль максимальной ошибки = {max_abs_error(relaxation_solution, analytical_solution)}')
    print(f'Модуль средней ошибки = {mean_abs_error(relaxation_solution, analytical_solution)}')
    print(f'Количество итераций = {relaxation_iters}')
    print()

    plot_combined_results_and_errors(
        solutions=solutions_3,
        analytical_solution_name=SolutionMethod.exact,
        x_range=(x_begin, x_end),
        y_range=(y_begin, y_end),
        h_x=h_x,
        h_y=h_y,
    )


if __name__ == "__main__":
    main()
