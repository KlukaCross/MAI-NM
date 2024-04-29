import math
import numpy as np
from typing import Callable

INF = 10e9


def vec_norm(v: np.ndarray) -> int:
    res = -INF
    for a in v:
        res = max(res, abs(a))
    return res


def mat_norm(m: np.ndarray) -> int:
    res = -INF
    for row in m:
        res = max(res, sum(map(abs, row)))
    return res


def get_jacobi_function(
    f1_dx1: Callable, f1_dx2: Callable, f2_dx1: Callable, f2_dx2: Callable
) -> Callable[[np.ndarray], np.ndarray]:
    def jacobi(x: np.ndarray) -> np.ndarray:
        x1 = x[0]
        x2 = x[1]
        return np.array(
            [
                [f1_dx1(x1, x2), f1_dx2(x1, x2)],
                [f2_dx1(x1, x2), f2_dx2(x1, x2)]
            ]
        )
    return jacobi


def newton_method(
    f: Callable[[np.ndarray], np.ndarray], j: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, eps: float
) -> tuple[np.ndarray, int]:
    iter_count = 0
    dx = 1
    x_k = x0
    while eps < dx:
        x_k_next = x_k - np.dot(np.linalg.inv(j(x_k)), f(x_k))  # todo: должно вычисляться из f(x_k) + J(x_k)*dx_x = 0
        dx = vec_norm(x_k_next - x_k)
        x_k = x_k_next
        iter_count += 1
    return x_k, iter_count


def simple_iterations(
    phi: Callable[[np.ndarray], np.ndarray],
    phi_dx: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    eps: float
) -> tuple[np.ndarray, int]:
    # todo: q должна быть максимальная
    iter_count = 0
    q = mat_norm(phi_dx(x0))
    coef = q / (1 - q)
    x_k = x0
    dx = 10e9
    while eps < coef * dx:
        x_k_next = phi(x_k)
        dx = vec_norm(x_k_next - x_k)
        x_k = x_k_next
        iter_count += 1
    return x_k, iter_count


def main():
    eps = float(input())
    a = 1
    f1 = lambda x1, x2: x1 - math.cos(x2) - a
    f2 = lambda x1, x2: x2 - math.sin(x1) - a
    f1_dx1 = lambda x1, x2: 1
    f1_dx2 = lambda x1, x2: math.cos(x2)
    f2_dx1 = lambda x1, x2: -math.cos(x1)
    f2_dx2 = lambda x1, x2: 1
    phi1 = lambda x1, x2: math.cos(x2) + a
    phi2 = lambda x1, x2: math.sin(x1) + a
    phi1_dx1 = lambda x1, x2: 0
    phi1_dx2 = lambda x1, x2: -math.sin(x2)
    phi2_dx1 = lambda x1, x2: math.cos(x1)
    phi2_dx2 = lambda x1, x2: 0
    x0 = np.array(
        [1, 2]
    )

    j = get_jacobi_function(f1_dx1, f1_dx2, f2_dx1, f2_dx2)
    phi_dx = get_jacobi_function(phi1_dx1, phi1_dx2, phi2_dx1, phi2_dx2)
    f = lambda x: np.array([f1(x[0], x[1]), f2(x[0], x[1])])
    phi = lambda x: np.array([phi1(x[0], x[1]), phi2(x[0], x[1])])

    x, iter_count = simple_iterations(phi, phi_dx, x0, eps)
    print(f"simple iterations:\n\tx1={x[0]}\n\tx2={x[1]}\n\titer_count={iter_count}")

    x, iter_count = newton_method(f, j, x0, eps)
    print(f"newton method:\n\tx1={x[0]}\n\tx2={x[1]}\n\titer_count={iter_count}")


main()
