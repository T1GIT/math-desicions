from tasks import task_1, task_2, task_3, task_4

tasks = [task_1, task_2, task_3, task_4]


def run():
    while True:
        number = int(input("Input task number: "))
        tasks[number - 1].run()


if __name__ == '__main__':
    run()