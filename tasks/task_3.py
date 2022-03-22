import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from sympy import Eq, Expr, lambdify, parse_expr, solve, symbols

from utils.abstract_task import AbstractTask


class Task3(AbstractTask):
    u_raw: str
    i_raw: str
    p: int
    q: int
    i: int
    bounds: [int, int]

    u_point: [int, int]
    u_value: int

    _u_expr: Expr
    _i_expr: Expr

    def __init__(self):
        self.i_raw = 'x * p + y * q'

    def input(self, default: bool = False):
        if default:
            self.u_raw = '5 * ln(x - 3) + 8 * ln(y - 3)'
            self.p = 15
            self.q = 4
            self.i = 422
            self.bounds = [4, 4]
        else:
            self.u_raw = input('Input U(x, y): ')
            self.p = int(input('Input p:'))
            self.q = int(input('Input q:'))
            self.i = int(input('Input i:'))
            bounds = input('Input bounds <x y>: ')
            self.bounds = list(map(int, bounds)) if bounds != '' else [-self.Config.CHART_MAX for _ in range(2)]

    def output(self, accuracy: int = 2):
        print(f'''
        Функция u: {self.u_raw}
        Значение p: {self.p}
        Значение q: {self.q}
        Значение i: {self.i}
        
        Оптимальная точка: x={self.u_point[0]} y={self.u_point[1]}
        Значение в оптимальной точке: {self.u_value:.{accuracy}f}
        ''', )

    def chart(self):
        x, y = symbols('x y')
        u_lam = lambdify((x, y), self._u_expr)
        y_lam = lambdify(x, solve(Eq(self._i_expr, self.i), y)[0])

        x_surface = np.outer(np.linspace(
            self.bounds[0],
            self.Config.CHART_MAX,
            self.Config.SURFACE_DENSITY
        ), np.ones(self.Config.SURFACE_DENSITY))
        y_surface = np.outer(np.linspace(
            self.bounds[1],
            self.Config.CHART_MAX,
            self.Config.SURFACE_DENSITY
        ), np.ones(self.Config.SURFACE_DENSITY)).T
        u_surface = np.vectorize(u_lam)(x_surface, y_surface)
        x_contour = np.linspace(
            self.bounds[0],
            self.Config.CHART_MAX,
            self.Config.CONTOUR_DENSITY
        )
        y_contour = np.linspace(
            self.bounds[1],
            self.Config.CHART_MAX,
            self.Config.CONTOUR_DENSITY
        )
        u_contour = np.vectorize(u_lam)(x_contour, y_contour.reshape((-1, 1)))
        x_constraint = np.linspace(
            self.bounds[0],
            float(solve(Eq(self._i_expr, self.i), x)[0].evalf(subs={y: self.bounds[1]})),
            self.Config.CONSTRAINT_DENSITY
        )
        y_constraint = np.vectorize(y_lam)(x_constraint)
        u_constraint = np.vectorize(u_lam)(x_constraint, y_constraint)

        go.Figure(data=[
            go.Surface(x=x_surface, y=y_surface, z=u_surface),
            go.Scatter3d(x=[self.u_point[0]], y=[self.u_point[1]], z=[self.u_value]),
            go.Scatter3d(x=x_constraint, y=y_constraint, z=u_constraint, mode='lines')

        ]).show()

        go.Figure(data=[
            go.Contour(x=x_contour, y=y_contour, z=u_contour),
            go.Scatter(x=[self.u_point[0]], y=[self.u_point[1]]),
            go.Scatter(x=x_constraint, y=y_constraint, mode='lines')
        ]).show()

    def calc(self):
        self._parse_functions()
        u_lam = lambdify(symbols('x y'), self._u_expr)
        i_lam = lambdify(symbols('x y'), self._i_expr)

        res = minimize(
            lambda args: -u_lam(*args),
            np.zeros((1, 2)),
            constraints={'type': 'ineq', 'fun': lambda args: -i_lam(*args) + self.i},
            bounds=[(bound, np.inf) for bound in self.bounds])

        self.u_point = res.x.tolist()
        self.u_value = -res.fun

    def _parse_functions(self):
        x, y = symbols('x, y')
        u_scope = {"x": x, "y": y}
        self._u_expr = parse_expr(self.u_raw, local_dict=u_scope, evaluate=False)

        p, q = symbols('p q')
        i_scope = {"p": p, "q": q}
        i_expr = parse_expr(self.i_raw, local_dict=i_scope, evaluate=False)
        self._i_expr = i_expr.subs([(p, self.p), (q, self.q)])

    class Config:
        SURFACE_DENSITY = 100
        CONTOUR_DENSITY = 100
        CONSTRAINT_DENSITY = 100
        CHART_MAX = 100


def run():
    task = Task3()
    task.input(True)
    task.calc()
    task.chart()
    task.output()
