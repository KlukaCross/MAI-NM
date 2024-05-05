import math


def euler_method(f, g, l, r, h, y0, z0):
    n = int((l+r)/h)
    x_k = l
    y_k = y0
    z_k = z0
    x = [x_k]
    y = [y_k]
    z = [z_k]
    for i in range(1, n+1):
        x_k += h
        y_k += h * f(x[i-1], y[i-1], z[i-1])
        z_k += h + g(x[i-1], y[i-1], z[i-1])
        x.append(x_k)
        y.append(y_k)
        z.append(z_k)

    return x, y, z


def runge_kutta_method(f, g, l, r, h, y0, z0):
    n = int((l+r)/h)
    x_k = l
    y_k = y0
    z_k = z0
    x = [x_k]
    y = [y_k]
    z = [z_k]
    for i in range(1, n+1):
        K1 = h * f(x_k, y_k, z_k)
        L1 = h * g(x_k, y_k, z_k)
        K2 = h * f(x_k + h/2, y_k + K1/2, z_k + L1/2)
        L2 = h * g(x_k + h/2, y_k + K1/2, z_k + L1/2)
        K3 = h * f(x_k + h/2, y_k + K2/2, z_k + L2/2)
        L3 = h * g(x_k + h/2, y_k + K2/2, z_k + L2/2)
        K4 = h * f(x_k + h, y_k + K3, z_k + K3)
        L4 = h * g(x_k + h, y_k + K3, z_k + K3)
        dy = (K1 + 2*K2 + 2*K3 + K4)/6
        dz = (L1 + 2*L2 + 2*L3 + L4)/6
        x_k += h
        y_k += dy
        z_k += dz
        x.append(x_k)
        y.append(y_k)
        z.append(z_k)
    return x, y, z


def main():
    f = lambda x, y, z: z
    g = lambda x, y, z: -z*math.tan(x) - y*math.cos(x)**2
    y0 = 0
    z0 = 0
    l, r = 0, 1
    h = 0.1

    euler_values = euler_method(f, g, l, r, h, y0, z0)


if __name__ == "__main__":
    main()
