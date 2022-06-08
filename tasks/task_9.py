import re
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from utils.abstract_task import AbstractTask


class Task9(AbstractTask):
    FLOAT__PATTERN = re.compile('^[+-]?\d(>?\.?\d+)?$')
    METHODS = {'keyboard', 'random', 'csv'}

    a: list[str]
    b: list[str]
    m: np.ndarray

    a_win: str
    a_price: int
    b_win: str
    b_price: int
    a_mixed_win: np.ndarray
    a_mixed_price: int
    b_mixed_win: np.ndarray
    b_mixed_price: int


    def _input_method(self) -> Literal['keyboard', 'random', 'csv']:
        method = None
        while method not in self.METHODS:
            method = input(f'Введите метод [{self.METHODS}]: ').strip().lower()
        return method

    def _input_keyboard(self):
        self.a = input('Введите стратегии игрока A: ').strip().split()
        self.b = input('Введите стратегии игрока B: ').strip().split()
        m = []
        for i in range(len(self.a)):
            while True:
                row = input(f'Введите {i} строка [{len(self.b)} чисел]: ').strip().split()
                if len(row) != len(self.b):
                    print(f'Требуется {len(self.b)} чисел')
                if all(map(lambda s: self.FLOAT__PATTERN.match(s), row)):
                    row = list(map(float, row))
                    break
            m.append(row)
        self.m = np.array(m)

    def _input_random(self):
        while True:
            a_len = input('Введите количество стратегий игрока А: ').strip()
            if a_len.isdigit():
                a_len = int(a_len)
                break
        while True:
            b_len = input('Введите количество стратегий игрока B: ').strip()
            if b_len.isdigit():
                b_len = int(b_len)
                break
        self.a = list(map(str, range(a_len)))
        self.b = list(map(str, range(b_len)))
        self.m = np.random.randint(0, 1e5, (a_len, b_len))


    def _input_csv(self):
        data = pd.read_csv('../data/task_9/strategies.csv')
        self.a = data['name'].tolist()
        self.b = data.columns.tolist()[1:]
        self.m = data.drop(['name'], axis=1).to_numpy()

    def input(self, default: bool = False):
        method = self._input_method()
        if method == 'keyboard':
            self._input_keyboard()
        elif method == 'random':
            self._input_random()
        elif method == 'csv':
            self._input_csv()

    def output(self):
        print(f'Оптимальная чистая стратегия для игрока А: {self.a_win}', end='\n\n')
        print(f'Цена игры для игрока А при выборе чистой оптимальной стратегии: {self.a_price}', end='\n\n')
        print(f'Оптимальная чистая стратегия для игрока B: {self.b_win}', end='\n\n')
        print(f'Цена игры для игрока B при выборе чистой оптимальной стратегии: {self.b_price}', end='\n\n')
        print('Таблица смешанных стратегий для игрока А')
        print(pd.DataFrame([self.a_mixed_win], columns=self.a).to_string(index=False), end='\n\n')
        print(f'Цена игры для игрока А при выборе смешанной оптимальной стратегии: {self.a_mixed_price}', end='\n\n')
        print('Таблица смешанных стратегий для игрока B')
        print(pd.DataFrame([self.b_mixed_win], columns=self.b).to_string(index=False), end='\n\n')
        print(f'Цена игры для игрока B при выборе смешанной оптимальной стратегии: {self.b_mixed_price}', end='\n\n')

    def calc(self):
        a_i = self.m.min(axis=1).argmax()
        self.a_win = self.a[a_i]
        self.a_price = round(self.m[a_i].min())
        b_i = self.m.max(axis=0).argmin()
        self.b_win = self.b[b_i]
        self.b_price = round(self.m[:,b_i].max())
        res = linprog(np.ones(len(self.a)), A_ub=-self.m.T, b_ub=-np.ones(len(self.b)))
        self.a_mixed_win = (res.x / res.x.sum() * 100).round().astype(int)
        self.a_mixed_price = round(1 / res.x.sum())
        res = linprog(np.ones(len(self.b)), A_ub=-self.m, b_ub=-np.ones(len(self.a)))
        self.b_mixed_win = (res.x / res.x.sum() * 100).round().astype(int)
        self.b_mixed_price = round(1 / res.x.sum())




def run():
    task = Task9()
    task.input(True)
    task.calc()
    task.output()


if __name__ == '__main__':
    run()
