from tasks import task_1, task_2, task_3

tasks = [task_1, task_2, task_3]


def run():
    while True:
        number = int(input("Input task number: "))
        tasks[number - 1].run()


if __name__ == '__main__':
    run()