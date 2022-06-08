import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from sympy import *

from utils.abstract_task import AbstractTask


class Task6(AbstractTask):
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
            self.is_max = True
            self.a_raw = ['1;3;4', '6;5;2']
            self.b_raw = ['3000', '3320']
            self.c_raw = '6 * 4 + 12;5 * 4 + 22;32'
        else:
            n = int(input('Input constraints amount: '))
            self.is_max = bool(int(input("Input method[1-max, 0-min]")))
            self.a_raw = [
                input(f"Input A [row {i}]")
                for i in range(n)
            ]
            self.b_raw = [
                input(f"Input B [row {i}]")
                for i in range(n)
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

    def calc(self):
        res = linprog(
            -self.c if self.is_max else self.c,
            A_ub=self.a,
            b_ub=self.b,
            bounds=[0, None]
        )

        self.point = list(map(round, res.x.tolist()))
        self.value = round((-1 if self.is_max else 1) * res.fun)


def run():
    task = Task6()
    task.input(True)
    task.calc()
    task.chart()
    task.output()


if __name__ == '__main__':
    run()
