from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def simple_solution():
    consumption = np.array([
        [7, 26, 95],
        [18, 24, 131]
    ])

    change = np.array([
        0.5,
        -0.2
    ])

    size = len(consumption)
    C = consumption[:, :size]
    Y = consumption[:, -1]
    X = consumption.sum(axis=1)
    A = C / X[:, None]
    H = np.linalg.inv(np.eye(size) - A)
    Y1 = Y * (1 + change)
    X1 = H @ Y1

    # index = np.arange(size)
    # plt.title('Change of consumption')
    # plt.axis([-0.5, size - 0.5, 0, max(X.max(), X1.max())])
    # plt.xticks(index, range(size))
    # plt.bar(index, X, color='pink')
    # plt.bar(index, X1 - X, color='#908493', bottom=X)
    # plt.show()


def read_amount_from_console() -> int:
    while True:
        n = input("Введите количество отраслей: ").strip()
        if not n.isdigit():
            print("Некорректное число")
            continue
        n = int(n)
        break
    return n


def read_consumption_from_console(n: int) -> np.ndarray:
    consumption = []
    for i in range(n):
        while True:
            row = input(f"Введите строку(X{i + 1}1 X{i + 1}2 ... X{i + 1}n X{i + 1}n+1): ").strip()
            row_split = row.split(" ")
            if len(row_split) != (n + 1):
                print("Неверное количество аргументов")
                continue
            try:
                consumption.append(list(map(float, row_split)))
                break
            except ValueError:
                print("Некорректное число")
                continue
    return np.array(consumption)


def read_change_from_console(n: int) -> np.ndarray:
    while True:
        row = input("Введите вектор изменений(Xn1 Xn2 ... Xnn Xnn+1): ").strip()
        row_split = row.split(" ")
        if len(row_split) != n:
            print("Неверное количество аргументов")
            continue
        try:
            change = list(map(lambda c: float(c) / 100, row_split))
            break
        except ValueError:
            print("Некорректное число")
            continue
    return np.array(change).reshape((-1, 1))


def read_consumption_from_csv() -> np.ndarray:
    df = pd.read_csv('./data/task_1/consumption.csv', sep=';', decimal='.', header=None, index_col=None)
    return df.to_numpy()


def read_change_from_csv() -> np.ndarray:
    df = pd.read_csv('./data/task_2/change.csv', sep=';', decimal='.', header=None, index_col=None)
    return df.to_numpy() / 100


def read_consumption_from_random(n: int) -> np.ndarray:
    return np.random.randint(0, 100, (n, n + 1))


def read_change_from_random(n: int) -> np.ndarray:
    return np.random.random((n, 1)) * 2 - 1


def calc_production(consumption: np.ndarray) -> np.ndarray:
    return consumption.sum(axis=1).reshape(-1, 1)


def calc_straight_costs(consumption: np.ndarray) -> np.ndarray:
    size = len(consumption)
    C = consumption[:, :size]
    X = calc_production(consumption)
    return C[:, :size] / X.T


def calc_full_costs(consumption: np.ndarray) -> np.ndarray:
    size = len(consumption)
    A = calc_straight_costs(consumption)
    return np.linalg.inv(np.eye(size) - A)


def calc_final_consumption(consumption: np.ndarray, change: np.ndarray) -> np.ndarray:
    Y = consumption[:, -1]
    return Y.reshape((-1, 1)) * (1 + change)


def calc_gross_production(consumption: np.ndarray, change: np.ndarray) -> np.ndarray:
    H = calc_full_costs(consumption)
    Y1 = calc_final_consumption(consumption, change)
    return H @ Y1


def calc_delta_gross_production(consumption: np.ndarray, change: np.ndarray) -> np.ndarray:
    X = calc_production(consumption)
    X1 = calc_gross_production(consumption, change)
    return (X1 - X) / X * 100


def calc_clear_production(consumption: np.ndarray, change: np.ndarray) -> np.ndarray:
    A = calc_straight_costs(consumption)
    X1 = calc_gross_production(consumption, change)
    return X1 - X1 * A.sum(axis=0).reshape(-1, 1)


def show_chart_gross_production_with_growth(consumption: np.ndarray, change: np.ndarray):
    X = calc_production(consumption).T[0]
    X1 = calc_gross_production(consumption, change).T[0]
    size = len(consumption)
    index = np.arange(size)
    plt.title('Change of consumption')
    plt.axis([-0.5, size - 0.5, 0, max(X.max(), X1.max())])
    plt.xticks(index, range(size))
    plt.bar(index, X, color='green', alpha=0.5)
    plt.bar(index, X1 - X, color='purple', alpha=0.3, bottom=X)
    plt.show()


def show_chart_delta_gross_production(consumption: np.ndarray, change: np.ndarray):
    X = calc_production(consumption).T[0]
    X1 = calc_gross_production(consumption, change).T[0]
    data = np.abs((X1 - X) / X * 100)
    size = len(consumption)
    index = np.arange(size)
    plt.title('Delta gross production')
    plt.axis([-0.5, size - 0.5, 0, max(data)])
    plt.xticks(index, range(size))
    plt.bar(index, data, color='green', alpha=0.5)
    plt.show()


def is_matrix_effective(consumption: np.ndarray):
    H = calc_full_costs(consumption)
    return np.all(H > 0)


def calc_forbenius_values(consumption: np.ndarray):
    A = calc_straight_costs(consumption)
    v = np.linalg.eigvals(A)
    return v.max()


def calc_forbenius_vector(consumption: np.ndarray):
    A = calc_straight_costs(consumption)
    w, v = np.linalg.eig(A)
    return np.abs(v[:, 1])


def print_array(title: str, array: np.ndarray, last: Optional[str] = None):
    df = pd.DataFrame(array)
    df.index = list(map(lambda x: f'Отрасль {x}', range(df.shape[0])))
    df.columns = list(map(lambda x: f'Отрасль {x}', range(df.shape[1])))
    if last:
        df.columns = [*df.columns[:-1], last]
    print(f"\n {title} \n", df)


def run(auto: bool = False):
    while True:
        if auto:
            input_type = 1
        else:
            input_type = input('Введите, каким способом вы хотите считать данные(0 - console, 1 - csv, 2 - random): ').strip()
        if input_type == '0':
            n = read_amount_from_console()
            consumption = read_consumption_from_console(n)
            change = read_change_from_console(n)
        elif input_type == '1':
            consumption = read_consumption_from_csv()
            change = read_change_from_csv()
        elif input_type == '2':
            n = read_amount_from_console()
            consumption = read_consumption_from_random(n)
            change = read_change_from_random(n)
        else:
            print('Incorrect type')
            continue

        print_array('Балансовая таблица', consumption, 'Y0')
        print_array('Валовый выпуск [X]', calc_production(consumption), 'X')
        print_array('Прямые затраты [A]', calc_straight_costs(consumption))
        print_array('Полные затраты [H]', calc_full_costs(consumption))
        print_array('Вектор изменений конечного потребления', change, 'DELTA')
        print_array('Конечное потребление [Y1]', calc_final_consumption(consumption, change), 'Y1')
        print_array('Валовый выпуск [X1]', calc_gross_production(consumption, change), 'X1')
        print_array('Изменение валового выпуска (%)', calc_delta_gross_production(consumption, change), 'DELTA')
        print_array('Вектор чистой продукции', calc_clear_production(consumption, change), 'XC')
        show_chart_gross_production_with_growth(consumption, change)
        show_chart_delta_gross_production(consumption, change)
        print('\n Продуктивность матрицы \n', is_matrix_effective(consumption))
        print('\n Число Форбениуса: \n', calc_forbenius_values(consumption))
        print_array('Вектор Форбениуса', calc_forbenius_vector(consumption))

        break
