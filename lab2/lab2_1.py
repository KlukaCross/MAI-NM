import math


def simple_iterations(f, phi, phi_d, eps, start_a, start_b):
    q = max(abs(phi_d(start_a)), abs(phi_d(start_b)))
    coef = q / (1 - q)
    x_k = 1
    x_k_next = 2
    while eps < coef * abs(x_k * x_k_next):
        pass


def main():
    eps = int(input())
    f = lambda x: math.sin(x) - 2*x**2 + 0.5
    phi = lambda x: math.sqrt((math.sin(x) + 0.5) / 2)
    start_a, start_b = 1, 2
