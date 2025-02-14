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


AvailableSolutionSchemas = [SolutionSchema.explicit, SolutionSchema.implicit, SolutionSchema.combined]


class ApproximatingBoundaryConditionsType(StrEnum):
    t_2t1p = "2t1p"
    t_3t2p = "3t2p"
    t_2t2p = "2t2p"


class ParabolicPartialDifferentialEquationSolver:
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
        self.sigma = self.tau / self.h**2

        if self.schema == SolutionSchema.implicit:
            self.theta = 1
        elif self.schema == SolutionSchema.explicit:
            self.theta = 0
            self.check_stability()
        else:
            self.theta = 0.5

        self.x = x
        self.t = t

    def phi_0(self, t):  # u_x(0, t)
        return np.exp(-0.5*t)

    def phi_l(self, t):  # u_x(pi, t)
        return -np.exp(-0.5*t)

    def psi(self, x):  # u(x, 0)
        return np.sin(x)

    def g(self, x, t):
        return 0.5 * np.exp(-0.5 * t) * np.sin(x)

    def exact_solution(self, x, t):
        return np.exp(-0.5*t)*np.sin(x)

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
        self.trd_d[0] = (2*self.sigma*self.theta*self.h*self.phi_0(self.t[k]) -
                         (uk[1] + self.theta * self.tau * self.g(self.x[1], self.t[k])) -
                         ((1 - self.theta)*self.sigma*(uk[0] - 2*uk[1] + uk[2] + self.h**2 * self.g(self.x[1], self.t[k-1]))))

    def fill_trd_0_2t2p(self, k, uk):
        self.trd_a[0] = 0
        self.trd_b[0] = -(2*self.sigma*self.theta + 1)
        self.trd_c[0] = 2*self.sigma*self.theta
        self.trd_d[0] = (2*self.sigma*self.theta*self.h*self.phi_0(self.t[k]) -
                         (uk[0] + self.theta*self.tau*self.g(self.x[0], self.t[k])) -
                         (2*(1 - self.theta)*self.sigma*(uk[1] - uk[0] - self.h*(self.phi_0(self.t[k-1])) + 0.5*self.h**2*self.g(self.x[0], self.t[k-1]))))

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
        self.trd_d[-1] = (2*self.sigma*self.theta*self.h*self.phi_l(self.t[k]) +
                          (uk[-2] + self.theta * self.tau * self.g(self.x[-2], self.t[k])) +
                          ((1 - self.theta)*self.sigma*(uk[-3] - 2*uk[-2] + uk[-1] + self.h**2 * self.g(self.x[-2], self.t[k-1]))))

    def fill_trd_l_2t2p(self, k, uk):
        self.trd_a[-1] = 2*self.sigma*self.theta
        self.trd_b[-1] = -(2*self.sigma*self.theta + 1)
        self.trd_c[-1] = 0
        self.trd_d[-1] = (-2*self.sigma*self.theta*self.h*self.phi_l(self.t[k]) -
                          (uk[-1] + self.theta*self.tau*self.g(self.x[-1], self.t[k])) -
                          (2 * (1-self.theta)*self.sigma*(uk[-2] - uk[-1] + self.h*self.phi_l(self.t[k-1]) + 0.5*self.h**2*self.g(self.x[-1], self.t[k-1]))))

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
        return -2*self.sigma*self.h*self.phi_0(self.t[k-1]) + 2*self.sigma*uk_prev[1] + (1 - 2*self.sigma)*uk_prev[0] + self.tau * self.g(self.x[0], self.t[k-1])

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
        return 2*self.sigma*self.h*self.phi_l(self.t[k-1]) + 2*self.sigma*uk_prev[-2] + (1 - 2*self.sigma)*uk_prev[-1] + self.tau * self.g(self.x[-1], self.t[k-1])

    def get_explicit_l(self, k, uk_prev, uk_cur):
        match self.boundary:
            case ApproximatingBoundaryConditionsType.t_2t1p:
                return self.get_explicit_l_2t1p(k, uk_prev, uk_cur)
            case ApproximatingBoundaryConditionsType.t_3t2p:
                return self.get_explicit_l_3t2p(k, uk_prev, uk_cur)
            case ApproximatingBoundaryConditionsType.t_2t2p:
                return self.get_explicit_l_2t2p(k, uk_prev, uk_cur)

    def solve_explicit(self):
        u = np.zeros((self.K, self.N))
        u[0] = self.psi(self.x)

        for k in range(1, self.K):
            for j in range(1, self.N - 1):
                u[k][j] = self.sigma*u[k-1][j-1] + (1 - 2*self.sigma)*u[k-1][j] + self.sigma*u[k-1][j+1] + self.tau * self.g(self.x[j], self.t[k-1])
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
            for j in range(1, self.N - 1):
                self.trd_d[j] = -(u[k - 1][j] + self.theta * self.tau * self.g(self.x[j], self.t[k]) + (
                            1 - self.theta) * self.sigma * (u[k - 1][j - 1] - 2 * u[k - 1][j] + u[k - 1][
                    j + 1] + self.h ** 2 * self.g(self.x[j], self.t[k - 1])))

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
                return self.solve_crank_nicholson()
            case SolutionSchema.combined:
                return self.solve_crank_nicholson()
            case SolutionSchema.exact:
                return self.solve_exact()

    def check_stability(self):
        assert self.sigma <= 0.5, self.sigma


def calc_max_abs_error(numeric, analytic):
    return np.abs(numeric - analytic).max(axis=1)


@click.command()
@click.option("--t", default=5., help="параметр сетки T")
@click.option("--l", default=np.pi, help="параметр сетки l")
@click.option("--start_n", default=4, help="параметр сетки N -- старт отсчета")
@click.option("--start_k", default=60, help="параметр сетки K -- старт отсчета")
@click.option("--end_n", default=9, help="параметр сетки N -- конец отсчета")
@click.option("--end_k", default=80, help="параметр сетки K -- конец отсчета")
@click.option("--grid_step", default=5, help="шаг прохода по мелкости разбиения сетки")
@click.option("--schema", default="all", help="схема решения: explicit - явная; implicit - неявная; combined - Кранка-Николсона; all (по умолчанию) - все схемы")
@click.option("--boundary", default="all", help="тип аппроксимации граничных условий: 2t1p - двухточечная первого порядка; 3t2p - трёхточечная второго порядка; 2t2p - двухточечная второго порядка; all (по умолчанию) - все виды")
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
    h = l / (n - 1)
    tau = t / (k-1)
    tt = [i * tau for i in range(k - 1)]
    tt.append(t)
    tt = np.array(tt)
    x = [i * h for i in range(n - 1)]
    x.append(l)
    x = np.array(x)

    results: list[tuple[SolutionSchema, ApproximatingBoundaryConditionsType, ...]] = []
    for sc, bndr in product(schemas, boundaries):
        solver = ParabolicPartialDifferentialEquationSolver(
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

    solver = ParabolicPartialDifferentialEquationSolver(
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
            h_tau_params.append(f"{np.log(h):,.3f} | {np.log(tau):,.3f}")
            tt = [i * tau for i in range(k - 1)]
            tt.append(t)
            tt = np.array(tt)
            x = [i * h for i in range(n - 1)]
            x.append(l)
            x = np.array(x)

            solver = ParabolicPartialDifferentialEquationSolver(
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

            solver = ParabolicPartialDifferentialEquationSolver(
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

        ax3.plot(h_tau_params, np.log(ers), label=f"{sc}_{bndr}")
    ax3.set_ylabel("Ошибка")
    ax3.set_xlabel("log(h) | log(tau)")
    ax3.grid()
    ax3.legend(loc='best')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()