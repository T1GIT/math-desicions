import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from sympy import *

from utils.abstract_task import AbstractTask


class Task1(AbstractTask):
    price_raw: list[str]
    req_raw: str
    storage_raw: str

    price: np.ndarray
    req: np.ndarray
    storage: np.ndarray

    point: [float, float]
    value: float

    def input(self, default: bool = False):
        if default:
            self.price_raw = ['6;100000000;15;4', '9;15;2;100000000', '6;12;7;1']
            self.req_raw = '30;80;60;110'
            self.storage_raw = '60;130;90'
        else:
            n = int(input('Input providers amount: '))
            self.price_raw = [
                input(f"Input providers [row {i}]")
                for i in range(n)
            ]
            self.req_raw = input("Input requirements: ")
            self.storage_raw = input("Input storage: ")
        self.price = np.array([
            list(map(float, a.split(';')))
            for a in self.price_raw
        ])

        self.req = np.array(list(map(float,self.req_raw.split(';'))))
        self.storage = np.array(list(map(float, self.storage_raw.split(';'))))

    def output(self, accuracy: int = 2):
        print('Оптимальный план')
        print(self.point)
        print('Оптимальное значение')
        print(self.value)

    def calc(self):
        m, n = self.price.shape
        eq = np.zeros(m * n * m).reshape(m, m * n)
        for p in range(m):
            eq[p, range(p * n, p * n + n)] = 1
        ub = np.zeros(n * m * n).reshape(n, m * n)
        for s in range(n):
            ub[s, range(s, m * n, n)] = 1

        res = linprog(
            self.price.flatten(),
            A_eq=eq,
            A_ub=ub,
            b_eq=self.storage,
            b_ub=self.req,
        )

        self.point = np.array(res.x.tolist()).round().reshape(self.price.shape).astype(int)
        self.value = round(res.fun)


def run():
    task = Task1()
    task.input(True)
    task.calc()
    task.chart()
    task.output()


if __name__ == '__main__':
    run()
