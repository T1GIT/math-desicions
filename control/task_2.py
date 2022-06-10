import re
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from utils.abstract_task import AbstractTask


class Task2(AbstractTask):
    a: np.ndarray

    a_win: np.ndarray
    a_price: int
    b_win: np.ndarray
    b_price: int

    def input(self, default: bool = False):
        if default:
            self.a = np.array([
                [10, 7],
                [8, 9],
                [12, 2]
            ])
        else:
            n = int(input('Input A amount'))
            a = []
            for i in range(n):
                a.append(list(map(int, input(f'Input {i} row').strip().split())))


    def output(self):
        print('Таблица смешанных стратегий для игрока А')
        print(pd.DataFrame([self.a_win]).to_string(index=False), end='\n\n')
        print(f'Цена игры для игрока А при выборе смешанной оптимальной стратегии: {self.a_price}', end='\n\n')
        print('Таблица смешанных стратегий для игрока B')
        print(pd.DataFrame([self.b_win]).to_string(index=False), end='\n\n')
        print(f'Цена игры для игрока B при выборе смешанной оптимальной стратегии: {self.b_price}', end='\n\n')

    def calc(self):
        res = linprog(np.ones(self.a.shape[0]), A_ub=-self.a.T, b_ub=-np.ones(self.a.shape[1]))
        self.a_win = (res.x / res.x.sum() * 100).round().astype(int)
        self.a_price = round(1 / res.x.sum())
        res = linprog(np.ones(self.a.shape[1]), A_ub=-self.a, b_ub=-np.ones(self.a.shape[0]))
        self.b_win = (res.x / res.x.sum() * 100).round().astype(int)
        self.b_price = round(1 / res.x.sum())


def run():
    task = Task2()
    task.input(True)
    task.calc()
    task.output()


if __name__ == '__main__':
    run()
