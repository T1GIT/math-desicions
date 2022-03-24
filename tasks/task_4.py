import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from sympy import Eq, Expr, lambdify, parse_expr, solve, symbols

from utils.abstract_task import AbstractTask


class Task3(AbstractTask):
    q_raw: str
    i_raw: str
    w1: int
    w2: int
    i: int
    bounds: [int, int]

    q_point: [int, int]
    q_value: int

    _q_expr: Expr
    _i_expr: Expr

    def __init__(self):
        self.q_raw = '30 * (x ** (1/2)) * (y ** (1/3))'
        self.i_raw = 'x * w1 + y * w2'

    def input(self, default: bool = False):
        if default:
            self.w1 = 5
            self.w2 = 10
            self.i = 600
            self.bounds = [1e-5, 1e-5]
        else:
            self.w1 = int(input('Input w1:'))
            self.w2 = int(input('Input w2:'))
            self.i = int(input('Input i:'))
            bounds = input('Input bounds <x y>: ')
            self.bounds = list(map(int, bounds)) if bounds != '' else [0, 0]

    def output(self, accuracy: int = 2):
        print(f'''
        Значение w1: {self.w1}
        Значение w2: {self.w2}
        Значение i: {self.i}
        
        Оптимальная точка: x={self.q_point[0]:.{accuracy}f} y={self.q_point[1]:.{accuracy}f}
        Значение в оптимальной точке: {self.q_value:.{accuracy}f}
        ''', )

    def chart(self):
        x, y = symbols('x y')
        q_lam = lambdify((x, y), self._q_expr)

        x_surface = np.outer(
            np.linspace(self.bounds[0], self.Config.MAX, self.Config.DENSITY),
            np.ones(self.Config.DENSITY))
        y_surface = np.outer(
            np.linspace(self.bounds[1], self.Config.MAX, self.Config.DENSITY),
            np.ones(self.Config.DENSITY)).T
        q_surface = np.vectorize(q_lam)(x_surface, y_surface)

        x_contour = np.linspace(self.bounds[0], self.Config.MAX, self.Config.DENSITY)
        y_contour = np.linspace(self.bounds[1], self.Config.MAX, self.Config.DENSITY)
        q_contour = np.vectorize(q_lam)(x_contour, y_contour.reshape((-1, 1)))

        y_lam = lambdify(x, solve(Eq(self._i_expr, self.i), y)[0])
        x_lam = lambdify(y, solve(Eq(self._i_expr, self.i), x)[0])
        x_min = max(x_lam(self.Config.MAX), self.bounds[0])
        x_max = min(x_lam(self.bounds[1]), self.Config.MAX)
        x_constraint = np.linspace(x_min, x_max, self.Config.DENSITY)
        y_constraint = np.vectorize(y_lam)(x_constraint)
        q_constraint = np.vectorize(q_lam)(x_constraint, y_constraint)

        y_lam = lambdify(x, solve(Eq(self._q_expr, self.q_value), y)[0])
        x_lam = lambdify(y, solve(Eq(self._q_expr, self.q_value), x)[0])
        x_min = max(x_lam(self.Config.MAX), self.bounds[0])
        x_max = min(x_lam(self.bounds[1]), self.Config.MAX)
        x_optimum = np.linspace(x_min, x_max, self.Config.DENSITY)
        y_optimum = np.vectorize(y_lam)(x_optimum)
        q_optimum = np.vectorize(q_lam)(x_optimum, y_optimum)

        go.Figure(data=[
            go.Surface(name="Простанственная модель", x=x_surface, y=y_surface, z=q_surface),
            go.Scatter3d(name="Оптимальная точка", x=[self.q_point[0]], y=[self.q_point[1]], z=[self.q_value], marker=dict(color='white')),
            go.Scatter3d(name="Ограничение", x=x_constraint, y=y_constraint, z=q_constraint, mode='lines'),
            go.Scatter3d(name="Кривая безразличия", x=x_optimum, y=y_optimum, z=q_optimum, mode='lines')
        ], ).show()

        go.Figure(data=[
            go.Contour(name="Простанственная модель", x=x_contour, y=y_contour, z=q_contour),
            go.Scatter(name="Оптимальная точка", x=[self.q_point[0]], y=[self.q_point[1]], marker=dict(color='white')),
            go.Scatter(name="Ограничение", x=x_constraint, y=y_constraint, mode='lines', fill="tozeroy"),
            go.Scatter(name="Кривая безразличия", x=x_optimum, y=y_optimum, mode='lines')
        ]).show()

    def calc(self):
        self._parse_functions()
        q_lam = lambdify(symbols('x y'), self._q_expr)
        i_lam = lambdify(symbols('x y'), self._i_expr)

        res = minimize(
            lambda args: -q_lam(*args),
            np.zeros((1, 2)),
            constraints={'type': 'ineq', 'fun': lambda args: -i_lam(*args) + self.i},
            bounds=[(bound, np.inf) for bound in self.bounds])

        self.q_point = res.x.tolist()
        self.q_value = -res.fun

    def _parse_functions(self):
        x, y = symbols('x, y')
        q_scope = {"x": x, "y": y}
        self._q_expr = parse_expr(self.q_raw, local_dict=q_scope, evaluate=False)

        w1, w2 = symbols('w1 w2')
        i_scope = {"w1": w1, "w2": w2}
        i_expr = parse_expr(self.i_raw, local_dict=i_scope, evaluate=False)
        self._i_expr = i_expr.subs([(w1, self.w1), (w2, self.w2)])

    class Config:
        DENSITY = 100
        MAX = 100


def run():
    task = Task3()
    task.input(True)
    task.calc()
    task.chart()
    task.output()
