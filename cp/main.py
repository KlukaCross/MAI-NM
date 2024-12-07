import math
import click

from lab3 import lab3_5 as lab3

INF = 1e9


def integrate_with_transforming_to_definite_integral_method(f, l, r, h, eps, calculate_method):
    def f_new(t):
        return (1. / t ** 2) * f(1. / t)

    result = 0
    if r == INF:
        new_r = max(eps, l)
        result += calculate_method(f_new, eps, 1. / new_r - eps, h)
    else:
        new_r = r

    if l == -INF:
        new_l = min(-eps, r)
        result += calculate_method(f_new, 1. / new_l + eps, -eps, h)
    else:
        new_l = l

    if new_l < new_r:
        result += calculate_method(f, new_l, new_r, h)
    return result


def integrate_with_limit_transition_method(f, l, r, h, eps):
    result = 0
    iters = 0

    if r == INF:
        finish = False
        cur_x = max(l, 0)
        while not finish:
            iters += 1
            new_result = result + h * f((cur_x + cur_x + h) * 0.5)
            cur_x += h
            if abs(new_result - result) < eps:
                finish = True
            result = new_result
    else:
        result += lab3.rect_method(f, 0, r, h)

    if l == -INF:
        finish = False
        cur_x = min(0, r)
        while not finish:
            iters += 1
            new_result = result + h * f((cur_x - h + cur_x) * 0.5)
            cur_x -= h
            if abs(new_result - result) < eps:
                finish = True
            result = new_result
    else:
        result += lab3.rect_method(f, l, 0, h)

    return result, iters


@click.command()
@click.option("--a", default=-INF, help="нижний предел интегрирования (по умолчанию -INF)")
@click.option("--b", default=INF, help="верхний предел интегрирования (по умолчанию INF)")
@click.option("--h", default=0.001, help="шаг h")
@click.option("--eps", default=1e-5, help="eps")
@click.option("--exact", default=math.pi / math.sqrt(2), help="точное решение")
def main(
    a,
    b,
    h,
    eps,
    exact
):
    def f(x):
        return 1 / (1 + x**4)

    res_definite_rect_method = integrate_with_transforming_to_definite_integral_method(f, a, b, h, eps, lab3.rect_method)
    res_definite_trapezoid_method = integrate_with_transforming_to_definite_integral_method(f, a, b, h, eps, lab3.trapezoid_method)
    res_definite_simpson_method = integrate_with_transforming_to_definite_integral_method(f, a, b, h, eps, lab3.simpson_method)
    res_limit, iters_limit = integrate_with_limit_transition_method(f, a, b, h, eps)

    print(f"""
Точное решение: {exact}

Интегрирование через замену переменных:
    Значение метода прямоугольников: {res_definite_rect_method} (ошибка {abs(exact-res_definite_rect_method)})
    Значение метода трапеций: {res_definite_trapezoid_method} (ошибка {abs(exact-res_definite_trapezoid_method)})
    Значение метода Симпсона: {res_definite_simpson_method} (ошибка {abs(exact-res_definite_simpson_method)})

Интегрирование через предельный переход (метод прямоугольников):
    Значение: {res_limit} (ошибка {abs(exact-res_limit)})
    Количество итераций: {iters_limit}
    """)


if __name__ == '__main__':
    main()
