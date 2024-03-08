import numpy as np

def tridiagonal_solve(matrix, d) -> np.ndarray:
    n = len(d)
    p = np.ndarray(n, dtype=float)
    q = np.ndarray(n, dtype=float)
    x = np.ndarray(n, dtype=float)
    a = np.array([0]+[matrix[i+1, i] for i in range(n-1)])
    b = np.array([matrix[i, i] for i in range(n)])
    c = np.array([matrix[i, i+1] for i in range(n-1)]+[0])

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        p[i] = -c[i] / (b[i] + a[i]*p[i-1])
        q[i] = (d[i] - a[i]*q[i-1]) / (b[i] + a[i]*p[i-1])

    x[-1] = q[-1]
    for i in range(n-2, -1, -1):
        x[i] = p[i] * x[i+1] + q[i]
    return x


def main():
    matrix = np.array([
        [-7, -6, 0, 0, 0],
        [6, 12, 0, 0, 0],
        [0, -3, 5, 0, 0],
        [0, 0, -9, 21, 8],
        [0, 0, 0, -4, -6]
    ], dtype=float)
    d = np.array([-75, 126, 13, -40, -24])
    x = tridiagonal_solve(matrix, d)
    print(x)

main()
