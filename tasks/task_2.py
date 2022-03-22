import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from plotly import subplots
from sympy import Expr, lambdify, parse_expr, symbols

from utils.abstract_task import AbstractTask


class Task2(AbstractTask):
    q1_raw: str
    q2_raw: str
    c_raw: str
    p_raw: str

    q_point: [int, int]
    p_point: [int, int]
    p_value: int

    _q1_expr: Expr
    _q2_expr: Expr
    _c_expr: Expr
    _p_expr: Expr

    def __init__(self):
        pass

    def input(self, default: bool = False):
        if default:
            self.q1_raw = '80 - 2 * p1'
            self.q2_raw = '58 - 1 * p2'
            self.c_raw = '2 * q1 ** 2 + 5 * q1 * q2 + 3 * q2 ** 2 + 7'
            self.p_raw = 'p1 * q1 + p2 * q2 - c'
        else:
            self.q1_raw = input('Input q1(p1): ')
            self.q2_raw = input('Input q2(p2): ')
            self.c_raw = input('Input c(q1, q2): ')
            self.p_raw = input('Input p(q1, q2): ')

    def output(self, accuracy: int = 2):
        print(f'''
        Функция q1: {self.q1_raw}
        Функция q2: {self.q2_raw}
        Функция c: {self.c_raw}
        Функция p: {self.p_raw}
        
        Оптимальная точка: q1={self.q_point[0]:.{accuracy}f} q2={self.q_point[1]:.{accuracy}f}
        Значение в оптимальной точке: {self.p_value:.{accuracy}f}
        ''', )

    def chart(self):
        p_lam = lambdify(symbols('p1 p2'), self._p_expr)

        p1_surface = np.outer(np.linspace(
            -self.Config.CHART_MAX,
            self.Config.CHART_MAX,
            self.Config.SURFACE_DENSITY
        ), np.ones(self.Config.SURFACE_DENSITY))
        p2_surface = p1_surface.T
        p_surface = np.vectorize(p_lam)(p1_surface, p2_surface)

        p1_contour = np.linspace(
            -self.Config.CHART_MAX,
            self.Config.CHART_MAX,
            self.Config.CONTOUR_DENSITY
        )
        p2_contour = p1_contour
        p_contour = np.vectorize(p_lam)(p1_contour, p2_contour.reshape((-1, 1)))

        go.Figure(data=[
            go.Surface(x=p1_surface, y=p2_surface, z=p_surface),
            go.Scatter3d(x=[self.p_point[0]], y=[self.p_point[1]], z=[self.p_value]),
        ]).show()

        go.Figure(data=[
            go.Contour(x=p1_contour, y=p2_contour, z=p_contour),
            go.Scatter(x=[self.p_point[0]], y=[self.p_point[1]])
        ]).show()

    def calc(self):
        self._parse_functions()
        p_lam = lambdify(symbols('p1 p2'), self._p_expr)
        res = minimize(lambda args: -p_lam(*args), np.zeros((1, 2)))

        self.p_point = res.x.tolist()
        self.p_value = -res.fun
        self.q_point = [
            self._q1_expr.subs(symbols('p1'), self.p_point[0]),
            self._q2_expr.subs(symbols('p2'), self.p_point[1])
        ]

    def _parse_functions(self):
        p1, p2 = symbols('p1 p2')
        q_scope = {"p1": p1, "p2": p2}
        self._q1_expr = parse_expr(self.q1_raw, local_dict=q_scope, evaluate=False)
        self._q2_expr = parse_expr(self.q2_raw, local_dict=q_scope, evaluate=False)

        q1, q2 = symbols('q1 q2')
        c_scope = {"q1": q1, "q2": q2}
        c_expr = parse_expr(self.c_raw, local_dict=c_scope, evaluate=False)
        self._c_expr = c_expr.subs([(q1, self._q1_expr), (q2, self._q2_expr)])

        c = symbols('c')
        p_scope = {**q_scope, **c_scope, "c": c}
        p_expr = parse_expr(self.p_raw, local_dict=p_scope, evaluate=False)
        self._p_expr = p_expr.subs([(q1, self._q1_expr), (q2, self._q2_expr), (c, self._c_expr)])

    class Config:
        SURFACE_DENSITY = 100
        CONTOUR_DENSITY = 100
        CHART_MAX = 100


def run():
    task = Task2()
    task.input(True)
    task.calc()
    task.chart()
    task.output()
