import numpy as np
import matplotlib.pyplot as plt


# consumption = np.array([
#     [7, 26, 95],
#     [18, 24, 131]
# ])
#
# change = np.array([
#     0.5,
#     -0.2
# ])
#
# size = len(consumption)
# C = consumption[:,:size]
# Y = consumption[:,-1]
# X = consumption.sum(axis=1)
# A = C / X[:, None]
# H = np.linalg.inv(np.eye(size) - A)
# Y1 = Y * (1 + change)
# X1 = H @ Y1
#
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
    for _ in range(n):
        while True:
            row = input("Введите строку(Xn1 Xn2 ... Xnn Xnn+1): ").strip()
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
    return np.array(change)


def calc_production(consumption: np.ndarray) -> np.ndarray:
    return consumption.sum(axis=1)


def calc_straight_costs(consumption: np.ndarray) -> np.ndarray:
    size = len(consumption)
    C = consumption[:, :size]
    X = calc_production(consumption)
    return C / X[:, None]


def calc_full_costs(consumption: np.ndarray) -> np.ndarray:
    size = len(consumption)
    A = calc_straight_costs(consumption)
    return np.linalg.inv(np.eye(size) - A)


def calc_final_consumption(consumption: np.ndarray, change: np.ndarray) -> np.ndarray:
    Y = consumption[:,-1]
    return Y * (1 + change)


def calc_gross_production(consumption: np.ndarray, change: np.ndarray) -> np.ndarray:
    H = calc_full_costs(consumption)
    Y1 = calc_final_consumption(consumption, change)
    return H @ Y1


def calc_delta_gross_production(consumption: np.ndarray, change: np.ndarray) -> np.ndarray:
    X = calc_production(consumption)
    X1 = calc_gross_production(consumption, change)
    return (X1 - X) / X * 100


def show_chart_with_growth(consumption: np.ndarray, change: np.ndarray) -> np.ndarray:
    X = calc_production(consumption)
    X1 = calc_gross_production(consumption, change)
    size = len(consumption)
    index = np.arange(size)
    plt.title('Change of consumption')
    plt.axis([-0.5, size - 0.5, 0, max(X.max(), X1.max())])
    plt.xticks(index, range(size))
    plt.bar(index, X, color='brown')
    plt.bar(index, X1 - X, color='gray', bottom=X)
    plt.show()


def run():
    n = read_amount_from_console()
    consumption = read_consumption_from_console(n)
    change = read_change_from_console(n)

    print("Валовый выпуск \n",
          calc_production(consumption))
    print("Прямые затраты \n",
          calc_straight_costs(consumption))
    print("Полные затраты \n",
          calc_full_costs(consumption))
    print("Конечное потребление \n",
          calc_final_consumption(consumption, change))
    print("Валовый выпуск \n",
          calc_gross_production(consumption, change))
    print("Изменение валового выпуска (%) \n",
          calc_delta_gross_production(consumption, change))

    show_chart_with_growth(consumption, change)


