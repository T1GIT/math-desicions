from tasks import task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9

tasks = [task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9]


def run():
    while True:
        number = int(input("Input task number: "))
        tasks[number - 1].run()


if __name__ == '__main__':
    run()