from enum import StrEnum
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import click

from lab1 import lab1_2 as lab1


class SolutionSchema(StrEnum):
    explicit = "explicit"
    implicit = "implicit"
    exact = "exact"


AvailableSolutionSchemas = [SolutionSchema.explicit, SolutionSchema.implicit]


class ApproximatingBoundaryConditionsType(StrEnum):
    t_2t1p = "2t1p"
    t_3t2p = "3t2p"
    t_2t2p = "2t2p"


class HyperbolicPartialDifferentialEquationSolver:
    def __init__(
        self,
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
        self.sigma = self.tau**2 / self.h**2

        if self.schema == SolutionSchema.explicit:
            self.check_stability()

        self.x = x
        self.t = t

    def psi_1(self, x):  # u(x, 0)
        return np.exp(2*x)

    def psi_2(self, x):  # u_t(x, 0)
        return 0

    def psi_1_dd(self, x):  # psi_1''(x, 0)
        return 4*self.psi_1(x)

    def exact_solution(self, x, t):
        return np.exp(2*x)*np.cos(t)

    def fill_first_two_u(self, u):
        u[0] = self.psi_1(self.x)
        u[1] = self.psi_1(self.x) + self.tau*self.psi_2(self.x) + self.tau**2*self.psi_1_dd(self.x)/2

    def clear_trd(self):
        self.trd_a = np.zeros(self.N)
        self.trd_b = np.zeros(self.N)
        self.trd_c = np.zeros(self.N)
        self.trd_d = np.zeros(self.N)

    def fill_trd_0_2t1p(self, k, u):
        self.trd_a[0] = 0
        self.trd_b[0] = (1 + 2*self.h)
        self.trd_c[0] = -1
        self.trd_d[0] = 0

    def fill_trd_0_3t2p(self, k, u):
        self.trd_a[0] = 0
        self.trd_b[0] = -(2 + 4*self.h)
        self.trd_c[0] = -(5*self.h**2 + 1/self.sigma - 2)
        self.trd_d[0] = (-2*u[k-1][1] + u[k-2][1])/self.sigma

    def fill_trd_0_2t2p(self, k, u):
        self.trd_a[0] = 0
        self.trd_b[0] = -(2 + 5*self.h**2 + 4*self.h + 1/self.sigma)
        self.trd_c[0] = 2
        self.trd_d[0] = (-2*u[k-1][0] + u[k-2][0])/self.sigma

    def fill_trd_0(self, k, u):
        match self.boundary:
            case ApproximatingBoundaryConditionsType.t_2t1p:
                self.fill_trd_0_2t1p(k, u)
            case ApproximatingBoundaryConditionsType.t_3t2p:
                self.fill_trd_0_3t2p(k, u)
            case ApproximatingBoundaryConditionsType.t_2t2p:
                self.fill_trd_0_2t2p(k, u)

    def fill_trd_l_2t1p(self, k, u):
        self.trd_a[-1] = -1
        self.trd_b[-1] = (1 - 2*self.h)
        self.trd_c[-1] = 0
        self.trd_d[-1] = 0

    def fill_trd_l_3t2p(self, k, u):
        self.trd_a[-1] = -(5*self.h**2 + 1/self.sigma - 2)
        self.trd_b[-1] = -(2 - 4*self.h)
        self.trd_c[-1] = 0
        self.trd_d[-1] = (-2*u[k-1][-2] + u[k-2][-2])/self.sigma

    def fill_trd_l_2t2p(self, k, u):
        self.trd_a[-1] = 2
        self.trd_b[-1] = -(2 + 5*self.h**2 - 4*self.h + 1/self.sigma)
        self.trd_c[-1] = 0
        self.trd_d[-1] = (-2*u[k-1][-1] + u[k-2][-1])/self.sigma

    def fill_trd_l(self, k, u):
        match self.boundary:
            case ApproximatingBoundaryConditionsType.t_2t1p:
                self.fill_trd_l_2t1p(k, u)
            case ApproximatingBoundaryConditionsType.t_3t2p:
                self.fill_trd_l_3t2p(k, u)
            case ApproximatingBoundaryConditionsType.t_2t2p:
                self.fill_trd_l_2t2p(k, u)

    def get_explicit_0_2t1p(self, k, u):
        return u[k][1] / (1 + 2*self.h)

    def get_explicit_0_3t2p(self, k, u):
        return (4*u[k][1] - u[k][2]) / (3 + 4*self.h)

    def get_explicit_0_2t2p(self, k, u):
        return self.sigma*(2*u[k-1][1] - (2 + 4*self.h)*u[k-1][0]) + (2 - 5*self.tau**2)*u[k-1][0] - u[k-2][0]

    def get_explicit_0(self, k, u):
        match self.boundary:
            case ApproximatingBoundaryConditionsType.t_2t1p:
                return self.get_explicit_0_2t1p(k, u)
            case ApproximatingBoundaryConditionsType.t_3t2p:
                return self.get_explicit_0_3t2p(k, u)
            case ApproximatingBoundaryConditionsType.t_2t2p:
                return self.get_explicit_0_2t2p(k, u)

    def get_explicit_l_2t1p(self, k, u):
        return u[k][-2] / (1 - 2*self.h)

    def get_explicit_l_3t2p(self, k, u):
        return (4*u[k][-2] - u[k][-3]) / (3 - 4*self.h)

    def get_explicit_l_2t2p(self, k, u):
        return self.sigma*(2*u[k-1][-2] + (4*self.h - 2)*u[k-1][-1]) + (2 - 5*self.tau**2)*u[k-1][-1] - u[k-2][-1]

    def get_explicit_l(self, k, u):
        match self.boundary:
            case ApproximatingBoundaryConditionsType.t_2t1p:
                return self.get_explicit_l_2t1p(k, u)
            case ApproximatingBoundaryConditionsType.t_3t2p:
                return self.get_explicit_l_3t2p(k, u)
            case ApproximatingBoundaryConditionsType.t_2t2p:
                return self.get_explicit_l_2t2p(k, u)

    def solve_explicit(self):
        u = np.zeros((self.K, self.N))
        self.fill_first_two_u(u)

        for k in range(2, self.K):
            for j in range(1, self.N - 1):
                u[k][j] = self.sigma*(u[k-1][j-1] - 2*u[k-1][j] + u[k-1][j+1]) - (5*self.tau**2*u[k-1][j]) + (2*u[k-1][j]) - u[k-2][j]
            u[k][0] = self.get_explicit_0(k, u)
            u[k][-1] = self.get_explicit_l(k, u)

        return u

    def solve_implicit(self):
        u = np.zeros((self.K, self.N))
        self.fill_first_two_u(u)

        for k in range(2, self.K):
            self.clear_trd()
            self.trd_a[:] = 1
            self.trd_b[:] = -(2 + 5*self.h**2 + 1/self.sigma)
            self.trd_c[:] = 1
            for j in range(1, self.N - 1):
                self.trd_d[j] = (u[k-2][j] - 2*u[k-1][j]) / self.sigma

            self.fill_trd_0(k, u)
            self.fill_trd_l(k, u)

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
            case SolutionSchema.exact:
                return self.solve_exact()

    def check_stability(self):
        assert self.sigma <= 1, self.sigma


def calc_max_abs_error(numeric, analytic):
    return np.abs(numeric - analytic).max(axis=1)


@click.command()
@click.option("--t", default=1., help="параметр сетки T")
@click.option("--l", default=1, help="параметр сетки l")
@click.option("--start_n", default=4, help="параметр сетки N -- старт отсчета")
@click.option("--start_k", default=50, help="параметр сетки K -- старт отсчета")
@click.option("--end_n", default=10, help="параметр сетки N -- конец отсчета")
@click.option("--end_k", default=100, help="параметр сетки K -- конец отсчета")
@click.option("--grid_step", default=7, help="шаг прохода по мелкости разбиения сетки")
@click.option("--schema", default="all", help="схема решения: explicit - явная; implicit - неявная; all (по умолчанию) - все схемы")
@click.option("--boundary", default="3t2p", help="тип аппроксимации граничных условий: 2t1p - двухточечная первого порядка; 3t2p - трёхточечная второго порядка; 2t2p - двухточечная второго порядка; all (по умолчанию) - все виды")
def main(
    t: int,
    l: int,
    start_n: int,
    start_k: int,
    end_n: int,
    end_k: int,
    grid_step: int,
    schema: str,
    boundary: str
):
    if schema != "all" and schema not in AvailableSolutionSchemas:
        print("bad schema parameter, see usage")
        return
    if boundary != "all" and boundary not in ApproximatingBoundaryConditionsType:
        print("bad boundary parameter, see usage")
        return

    if schema == "all":
        schemas = list(AvailableSolutionSchemas)
    else:
        schemas = [SolutionSchema[schema]]
    if boundary == "all":
        boundaries = list(ApproximatingBoundaryConditionsType)
    else:
        boundaries = [ApproximatingBoundaryConditionsType["t_"+boundary]]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    ax1.set_title("U(x)")
    n = end_n
    k = end_k
    fix_time = k-1
    h = l / (n-1)
    tau = t / (k-1)
    tt = [i * tau for i in range(k-1)]
    tt.append(t)
    tt = np.array(tt)
    x = [i * h for i in range(n-1)]
    x.append(l)
    x = np.array(x)

    results: list[tuple[SolutionSchema, ApproximatingBoundaryConditionsType, ...]] = []
    for sc, bndr in product(schemas, boundaries):
        solver = HyperbolicPartialDifferentialEquationSolver(
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
        results.append((sc, bndr, result))
        ax1.plot(x, result[fix_time], label=f"{sc}_{bndr}")

    solver = HyperbolicPartialDifferentialEquationSolver(
        T=t,
        l=l,
        N=n,
        K=k,
        schema=SolutionSchema.exact,
        boundary=boundaries[0],
        x=x,
        t=tt,
        tau=tau,
        h=h
    )
    result = solver.solve()
    exact_result = result
    ax1.plot(x, result[fix_time], label=str(SolutionSchema.exact))

    ax1.grid()
    ax1.legend(loc='best')
    ax1.set_ylabel("U")
    ax1.set_xlabel("x")

    ax2.set_title("График ошибки от времени")
    for sc, bndr, res in results:
        ax2.plot(tt, calc_max_abs_error(res, exact_result), label=f"{sc}_{bndr}")
    ax2.set_ylabel("Ошибка")
    ax2.set_xlabel("t")
    ax2.grid()
    ax2.legend(loc='best')

    ax3.set_title("График ошибки от параметров мелкости разбиения сетки")
    n_step = (end_n - start_n) // grid_step
    k_step = (end_k - start_k) // grid_step
    nn = [start_n + n_step*i for i in range(grid_step)]
    nn = np.array(nn)
    kk = [start_k + k_step*i for i in range(grid_step)]
    kk = np.array(kk)

    for sc, bndr in product(schemas, boundaries):
        ers = []
        h_tau_params = []
        for step in range(grid_step):
            n = nn[step]
            k = kk[step]
            h = l / (n - 1)
            tau = t / (k - 1)
            h_tau_params.append(f"{h:,.3f} | {tau:,.3f}")
            tt = [i * tau for i in range(k - 1)]
            tt.append(t)
            tt = np.array(tt)
            x = [i * h for i in range(n - 1)]
            x.append(l)
            x = np.array(x)

            solver = HyperbolicPartialDifferentialEquationSolver(
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

            solver = HyperbolicPartialDifferentialEquationSolver(
                T=t,
                l=l,
                N=n,
                K=k,
                schema=SolutionSchema.exact,
                boundary=bndr,
                x=x,
                t=tt,
                tau=tau,
                h=h
            )
            exact_result = solver.solve()
            ers.append(max(calc_max_abs_error(result, exact_result)))

        ax3.plot(h_tau_params, ers, label=f"{sc}_{bndr}")
    ax3.set_ylabel("Ошибка")
    ax3.set_xlabel("h | tau")
    ax3.grid()
    ax3.legend(loc='best')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()