import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from sympy import *

from utils.abstract_task import AbstractTask


class Task5(AbstractTask):
    n: int
    is_max: bool
    a_raw: list[str]
    b_raw: list[str]
    c_raw: str

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    point: [float, float]
    value: float

    def input(self, default: bool = False):
        if default:
            self.n = 4
            self.is_max = True
            self.a_raw = ['1;2', '3;2', '-1;1', '(3/4);-1']
            self.b_raw = ['10', '18', '8', '8 * (3/4) + 3']
            self.c_raw = '1;4'
        else:
            self.n = int(input('Input n: '))
            self.is_max = bool(int(input("Input method[1-max, 0-min]")))
            self.a_raw = [
                input(f"Input A [row {i}]")
                for i in range(self.n)
            ]
            self.b_raw = [
                input(f"Input B [row {i}]")
                for i in range(self.n)
            ]
            self.c_raw = input("Input C")
        self.a = np.array([
            list(map(float, map(sympify, a.split(';'))))
            for a in self.a_raw
        ])
        self.b = np.array([
            list(map(float, map(sympify, b.split(';'))))
            for b in self.b_raw
        ])
        self.c = np.array(list(map(float, map(sympify, self.c_raw.split(';')))))

    def output(self, accuracy: int = 2):
        print(f'''
        Точка: {self.point}
        Значение: {self.value}
        ''', )

    def chart(self):
        fig = go.Figure(layout=go.Layout(plot_bgcolor='rgb(200, 200, 255)'))
        alpha = 1 / self.a.shape[0]
        for [a, b, c] in np.concatenate([self.a, -self.b], axis=1):
            x = np.array([-self.Config.RANGE, self.Config.RANGE])
            y = np.vectorize(lambda x: -(c - a * -x) / b)(x)
            fig.add_trace(go.Scatter(name=f'{a};{b};{c}', x=x, y=y, line_color='red', fill='tozeroy', fillcolor=f'rgba(255, 100, 100, {alpha})'))

        x = np.array([-self.Config.RANGE, self.Config.RANGE])
        y = np.vectorize(lambda x: (self.value - (self.c[0] * x)) / self.c[1])(x)
        fig.add_trace(go.Scatter(name=f'Целевая', x=x, y=y, line_color='purple'))

        offset = -10 if self.is_max else 10
        x = np.array([-self.Config.RANGE, self.Config.RANGE])
        y = np.vectorize(lambda x: (self.value + offset - (self.c[0] * x)) / self.c[1])(x)
        fig.add_trace(go.Scatter(name=f'Смещённая', x=x, y=y, line_color='brown'))

        fig.add_trace(go.Scatter(name='Оптимум', x=[self.point[0]], y=[self.point[1]], line_color='green'))
        fig.show()

    def calc(self):
        res = linprog(
            -self.c if self.is_max else self.c,
            A_ub=self.a,
            b_ub=self.b,
            bounds=[-self.Config.RANGE, self.Config.RANGE]
        )

        self.point = list(map(round, res.x.tolist()))
        self.value = round((-1 if self.is_max else 1) * res.fun)

    class Config:
        RANGE = 10


def run():
    task = Task5()
    task.input(True)
    task.calc()
    task.chart()
    task.output()


if __name__ == '__main__':
    run()
