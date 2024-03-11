import numpy as np


def LU_decompose(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(A)
    swaps_count = 0
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    Ak = A.copy()
    for k in range(n-1):
        if Ak[k, k] == 0:
            for j in range(k+1, n):
                if Ak[j, k] != 0:
                    tmp = Ak[j, :].copy()
                    Ak[j, :] = Ak[k, :]
                    Ak[k, :] = tmp
                    swaps_count += 1
                    break

        U[k, :] = Ak[k, :]
        L[:, k] = Ak[:, k] / U[k, k]
        Ak -= np.outer(L[:, k], U[k, :])

    U[n-1, n-1] = Ak[n-1, n-1]
    det = (-1)**swaps_count
    for i in range(n):
        det *= U[i, i]
    return L, U, det


def solve(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    z = np.ndarray(n, dtype=float)
    x = np.ndarray(n, dtype=float)

    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= z[j] * L[i, j]
        z[i] = s

    for i in range(n-1, -1, -1):
        if U[i][i] == 0:
            continue
        s = z[i]
        for j in range(n-1, i, -1):
            s -= x[j] * U[i][j]
        x[i] = s / U[i][i]
    return x


def invert(L: np.ndarray, U: np.ndarray) -> np.ndarray:
    n = len(L)
    res = np.ndarray((n, n), dtype=float)
    for i in range(n):
        b = np.zeros(n)
        b[i] = 1.
        x = solve(L, U, b)
        for j in range(n):
            res[j, i] = x[j]
    return res


def main():
    n = int(input())
    A = np.array([list(map(int, input().split())) for _ in range(n)], dtype=float)
    b = np.array(list(map(int, input().split())))
    L, U, det = LU_decompose(A)
    x = solve(L, U, b)
    invert_matrix = invert(L, U)
    print(f"x: {x}")
    print(f"det: {det}")
    print(f"invert_matrix:\n{invert_matrix}")


main()
