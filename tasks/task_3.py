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
            self.bounds = list(map(int, bounds)) if bounds != '' else [-self.Config.MAX for _ in range(2)]

    def output(self, accuracy: int = 2):
        print(f'''
        Функция u: {self.u_raw}
        Значение p: {self.p}
        Значение q: {self.q}
        Значение i: {self.i}
        
        Оптимальная точка: x={self.u_point[0]:.{accuracy}f} y={self.u_point[1]:.{accuracy}f}
        Значение в оптимальной точке: {self.u_value:.{accuracy}f}
        ''', )

    def chart(self):
        x, y = symbols('x y')
        u_lam = lambdify((x, y), self._u_expr)

        x_surface = np.outer(
            np.linspace(self.bounds[0], self.Config.MAX, self.Config.DENSITY),
            np.ones(self.Config.DENSITY))
        y_surface = np.outer(
            np.linspace(self.bounds[1], self.Config.MAX, self.Config.DENSITY),
            np.ones(self.Config.DENSITY)).T
        u_surface = np.vectorize(u_lam)(x_surface, y_surface)

        x_contour = np.linspace(self.bounds[0], self.Config.MAX, self.Config.DENSITY)
        y_contour = np.linspace(self.bounds[1], self.Config.MAX, self.Config.DENSITY)
        u_contour = np.vectorize(u_lam)(x_contour, y_contour.reshape((-1, 1)))

        y_lam = lambdify(x, solve(Eq(self._i_expr, self.i), y)[0])
        x_lam = lambdify(y, solve(Eq(self._i_expr, self.i), x)[0])
        x_max = x_lam(self.bounds[0])
        x_constraint = np.linspace(self.bounds[0], x_max, self.Config.DENSITY)
        y_constraint = np.vectorize(y_lam)(x_constraint)
        u_constraint = np.vectorize(u_lam)(x_constraint, y_constraint)

        y_lam = lambdify(x, solve(Eq(self._u_expr, self.u_value), y)[0])
        x_lam = lambdify(y, solve(Eq(self._u_expr, self.u_value), x)[0])
        x_min = x_lam(self.Config.MAX)
        x_optimum = np.linspace(x_min, self.Config.MAX, self.Config.DENSITY)
        y_optimum = np.vectorize(y_lam)(x_optimum)
        u_optimum = np.vectorize(u_lam)(x_optimum, y_optimum)

        go.Figure(data=[
            go.Surface(name="Простанственная модель", x=x_surface, y=y_surface, z=u_surface),
            go.Scatter3d(name="Оптимальная точка", x=[self.u_point[0]], y=[self.u_point[1]], z=[self.u_value]),
            go.Scatter3d(name="Ограничение", x=x_constraint, y=y_constraint, z=u_constraint, mode='lines'),
            go.Scatter3d(name="Кривая безразличия", x=x_optimum, y=y_optimum, z=u_optimum, mode='lines')
        ], ).show()

        go.Figure(data=[
            go.Contour(name="Простанственная модель", x=x_contour, y=y_contour, z=u_contour),
            go.Scatter(name="Оптимальная точка", x=[self.u_point[0]], y=[self.u_point[1]]),
            go.Scatter(name="Ограничение", x=x_constraint, y=y_constraint, mode='lines', fill="tozeroy"),
            go.Scatter(name="Кривая безразличия", x=x_optimum, y=y_optimum, mode='lines')
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
        DENSITY = 100
        MAX = 100


def run():
    task = Task3()
    task.input(True)
    task.calc()
    task.chart()
    task.output()
