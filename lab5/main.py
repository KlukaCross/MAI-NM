from enum import StrEnum
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import click

from lab1 import lab1_2 as lab1


class SolutionSchema(StrEnum):
    explicit = "explicit"
    implicit = "implicit"
    combined = "combined"
    exact = "exact"


class ApproximatingBoundaryConditionsType(StrEnum):
    t_2t1p = "2t1p"
    t_3t2p = "3t2p"
    t_2t2p = "2t2p"


class ParabolicPartialDifferentialEquationSolver:
    def __init__(
        self,
        a: float,
        T: int,
        l: int,
        N: int,
        K: int,
        schema: SolutionSchema,
        boundary: ApproximatingBoundaryConditionsType,
        x,
        t,
        tau,
        h
    ):
        self.a = a
        self.T = T
        self.l = l
        self.N = N
        self.K = K
        self.schema = schema
        self.boundary = boundary

        self.trd_a = None
        self.trd_b = None
        self.trd_c = None
        self.trd_d = None

        self.h = h
        self.tau = tau
        self.sigma = self.a * self.tau / self.h**2
        self.check_stability()

        if self.schema == SolutionSchema.implicit:
            self.theta = 1
        elif self.schema == SolutionSchema.explicit:
            self.theta = 0
        else:
            self.theta = 0.5

        self.x = x
        self.t = t

    def phi_0(self, t):  # u_x(0, t)
        return np.exp(-self.a*t)

    def phi_l(self, t):  # u_x(pi, t)
        return -np.exp(-self.a*t)

    def psi(self, x):  # u(x, 0)
        return np.sin(x)

    def exact_solution(self, x, t):
        return np.exp(-self.a*t)*np.sin(x)

    def clear_trd(self):
        self.trd_a = np.zeros(self.N)
        self.trd_b = np.zeros(self.N)
        self.trd_c = np.zeros(self.N)
        self.trd_d = np.zeros(self.N)

    def fill_trd_0_2t1p(self, k, uk):
        self.trd_a[0] = 0
        self.trd_b[0] = -1
        self.trd_c[0] = 1
        self.trd_d[0] = self.h * self.phi_0(self.t[k])

    def fill_trd_0_3t2p(self, k, uk):
        self.trd_a[0] = 0
        self.trd_b[0] = -2*self.sigma*self.theta
        self.trd_c[0] = 2*self.sigma*self.theta - 1
        self.trd_d[0] = 2*self.sigma*self.theta*self.h*self.phi_0(self.t[k]) - (uk[1] + (1 - self.theta)*self.sigma*(uk[0] - 2*uk[1] + uk[2]))

    def fill_trd_0_2t2p(self, k, uk):
        self.trd_a[0] = 0
        self.trd_b[0] = -(2*self.sigma*self.theta + 1)
        self.trd_c[0] = 2*self.sigma*self.theta
        self.trd_d[0] = 2*self.sigma*self.theta*self.h*self.phi_0(self.t[k]) - uk[0] - (2*(1 - self.theta)*self.sigma*(uk[1] - uk[0] - self.h*(self.phi_0(self.t[k-1]))))

    def fill_trd_0(self, k, uk_prev):
        match self.boundary:
            case ApproximatingBoundaryConditionsType.t_2t1p:
                self.fill_trd_0_2t1p(k, uk_prev)
            case ApproximatingBoundaryConditionsType.t_3t2p:
                self.fill_trd_0_3t2p(k, uk_prev)
            case ApproximatingBoundaryConditionsType.t_2t2p:
                self.fill_trd_0_2t2p(k, uk_prev)

    def fill_trd_l_2t1p(self, k, uk):
        self.trd_a[-1] = -1
        self.trd_b[-1] = 1
        self.trd_c[-1] = 0
        self.trd_d[-1] = self.h * self.phi_l(self.t[k])

    def fill_trd_l_3t2p(self, k, uk):
        self.trd_a[-1] = 1 - 2*self.sigma*self.theta
        self.trd_b[-1] = 2*self.sigma*self.theta
        self.trd_c[-1] = 0
        self.trd_d[-1] = 2*self.sigma*self.theta*self.h*self.phi_l(self.t[k]) + (uk[-2] + (1 - self.theta)*self.sigma*(uk[-3] - 2*uk[-2] + uk[-1]))

    def fill_trd_l_2t2p(self, k, uk):
        self.trd_a[-1] = 2*self.sigma*self.theta
        self.trd_b[-1] = -(2*self.sigma*self.theta + 1)
        self.trd_c[-1] = 0
        self.trd_d[-1] = -2*self.sigma*self.theta*self.h*self.phi_l(self.t[k]) - uk[-1] - (2 * (1-self.theta)*self.sigma*(uk[-2] - uk[-1] + self.h*self.phi_l(self.t[k-1])))

    def fill_trd_l(self, k, uk_prev):
        match self.boundary:
            case ApproximatingBoundaryConditionsType.t_2t1p:
                self.fill_trd_l_2t1p(k, uk_prev)
            case ApproximatingBoundaryConditionsType.t_3t2p:
                self.fill_trd_l_3t2p(k, uk_prev)
            case ApproximatingBoundaryConditionsType.t_2t2p:
                self.fill_trd_l_2t2p(k, uk_prev)

    def get_explicit_0_2t1p(self, k, uk_prev, uk_cur):
        return -self.h * self.phi_0(self.t[k]) + uk_cur[1]

    def get_explicit_0_3t2p(self, k, uk_prev, uk_cur):
        return (-2*self.h*self.phi_0(self.t[k]) + 4*uk_cur[1] - uk_cur[2]) / 3

    def get_explicit_0_2t2p(self, k, uk_prev, uk_cur):
        return -2*self.sigma*self.h*self.phi_0(self.t[k-1]) + 2*self.sigma*uk_prev[1] + (1 - 2*self.sigma)*uk_prev[0]

    def get_explicit_0(self, k, uk_prev, uk_cur):
        match self.boundary:
            case ApproximatingBoundaryConditionsType.t_2t1p:
                return self.get_explicit_0_2t1p(k, uk_prev, uk_cur)
            case ApproximatingBoundaryConditionsType.t_3t2p:
                return self.get_explicit_0_3t2p(k, uk_prev, uk_cur)
            case ApproximatingBoundaryConditionsType.t_2t2p:
                return self.get_explicit_0_2t2p(k, uk_prev, uk_cur)

    def get_explicit_l_2t1p(self, k, uk_prev, uk_cur):
        return self.h * self.phi_l(self.t[k]) + uk_cur[-2]

    def get_explicit_l_3t2p(self, k, uk_prev, uk_cur):
        return (2*self.h*self.phi_l(self.t[k]) + 4*uk_cur[-2] - uk_cur[-3]) / 3

    def get_explicit_l_2t2p(self, k, uk_prev, uk_cur):
        return 2*self.sigma*self.h*self.phi_l(self.t[k-1]) + 2*self.sigma*uk_prev[-2] + (1 - 2*self.sigma)*uk_prev[-1]

    def get_explicit_l(self, k, uk_prev, uk_cur):
        match self.boundary:
            case ApproximatingBoundaryConditionsType.t_2t1p:
                return self.get_explicit_l_2t1p(k, uk_prev, uk_cur)
            case ApproximatingBoundaryConditionsType.t_3t2p:
                return self.get_explicit_l_3t2p(k, uk_prev, uk_cur)
            case ApproximatingBoundaryConditionsType.t_2t2p:
                return self.get_explicit_l_2t2p(k, uk_prev, uk_cur)

    def solve_implicit(self):
        u = np.zeros((self.K, self.N))
        u[0] = self.psi(self.x)

        for k in range(1, self.K):
            self.clear_trd()
            self.trd_a[:] = self.sigma * self.theta
            self.trd_b[:] = -1 - 2*self.sigma*self.theta
            self.trd_c[:] = self.sigma * self.theta
            self.trd_d[1:-1] = -u[k-1][1:-1]

            self.fill_trd_0(k, u[k-1])
            self.fill_trd_l(k, u[k-1])

            u[k][:] = lab1.tridiagonal_solve(self.trd_a, self.trd_b, self.trd_c, self.trd_d)

        return u

    def solve_explicit(self):
        u = np.zeros((self.K, self.N))
        u[0] = self.psi(self.x)

        for k in range(1, self.K):
            for j in range(1, self.N - 1):
                u[k][j] = self.sigma*u[k-1][j-1] + (1 - 2*self.sigma)*u[k-1][j] + self.sigma*u[k-1][j+1]
            u[k][0] = self.get_explicit_0(k, u[k-1], u[k])
            u[k][-1] = self.get_explicit_l(k, u[k-1], u[k])

        return u

    def solve_crank_nicholson(self):
        u = np.zeros((self.K, self.N))
        u[0] = self.psi(self.x)

        for k in range(1, self.K):
            self.clear_trd()
            self.trd_a[:] = self.sigma * self.theta
            self.trd_b[:] = -1 - 2*self.sigma*self.theta
            self.trd_c[:] = self.sigma * self.theta
            self.trd_d[1:-1] = np.array([-(u[k-1][i] + (1-self.theta) * self.sigma * (u[k-1][i-1] - 2*u[k-1][i] + u[k-1][i+1])) for i in range(1, self.N-1)])

            self.fill_trd_0(k, u[k-1])
            self.fill_trd_l(k, u[k-1])

            u[k][:] = lab1.tridiagonal_solve(self.trd_a, self.trd_b, self.trd_c, self.trd_d)

        return u

    def solve_exact(self):
        u = np.zeros((self.K, self.N))

        for k in range(self.K):
            for j in range(self.N):
                u[k][j] = self.exact_solution(self.x[j], self.t[k])
        return u

    def solve(self):
        match self.schema:
            case SolutionSchema.explicit:
                return self.solve_explicit()
            case SolutionSchema.implicit:
                return self.solve_implicit()
            case SolutionSchema.combined:
                return self.solve_crank_nicholson()
            case SolutionSchema.exact:
                return self.solve_exact()

    def check_stability(self):
        assert self.sigma <= 0.5, self.sigma


def calc_mean_abs_error(numeric, analytic):
    return np.abs(numeric - analytic).mean(axis=1)


@click.command()
@click.option("--a", default=0.5, help="коэффициент a")
@click.option("--t", default=5., help="параметр сетки T")
@click.option("--l", default=np.pi, help="параметр сетки l")
@click.option("--n", default=10, help="параметр сетки N")
@click.option("--k", default=70, help="параметр сетки K")
@click.option("--schema", default="all", help="схема решения: explicit - явная; implicit - неявная; combined - Кранка-Николсона; all (по умолчанию) - все схемы")
@click.option("--boundary", default="all", help="тип аппроксимации граничных условий: 2t1p - двухточечная первого порядка; 3t2p - трёхточечная второго порядка; 2t2p - двухточечная второго порядка; all (по умолчанию) - все виды")
def main(
    a: float,
    t: int,
    l: int,
    n: int,
    k: int,
    schema: str,
    boundary: str
):
    if schema != "all" and schema not in SolutionSchema:
        print("bad schema parameter, see usage")
        return
    if boundary != "all" and boundary not in ApproximatingBoundaryConditionsType:
        print("bad boundary parameter, see usage")
        return

    if schema == "all":
        schemas = list(SolutionSchema)
    else:
        schemas = [SolutionSchema[schema]]
    if boundary == "all":
        boundaries = list(ApproximatingBoundaryConditionsType)
    else:
        boundaries = [ApproximatingBoundaryConditionsType["t_"+boundary]]

    time = k-1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.set_title("U(x)")

    h = l / (n - 1)
    tau = t / (k-1)
    tt = [i * tau for i in range(k - 1)]
    tt.append(t)
    tt = np.array(tt)
    x = [i * h for i in range(n - 1)]
    x.append(l)
    x = np.array(x)

    results = []
    for sc, bndr in product(schemas, boundaries):
        solver = ParabolicPartialDifferentialEquationSolver(
            a=a,
            T=t,
            l=l,
            N=n,
            K=k,
            schema=sc,
            boundary=bndr,
            x=x,
            t=tt,
            tau=tau,
            h=h
        )
        result = solver.solve()
        results.append(result)
        ax1.plot(x, result[time], label=f"{sc}_{bndr}")

    ax1.grid()
    ax1.legend(loc='best')
    ax1.set_ylabel("U")
    ax1.set_xlabel("x")

    if schema == "all":
        ax2.set_title("График ошибки")
        exact_res = None
        for res, sc in zip(results, schemas):
            if sc == SolutionSchema.exact:
                exact_res = res
                break
        for res, sc in zip(results, schemas):
            if sc == SolutionSchema.exact:
                continue
            ax2.plot(tt, calc_mean_abs_error(res, exact_res), label=sc)
        ax2.set_ylabel("Ошибка")
        ax2.set_xlabel("t")
        ax2.grid()
        ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()