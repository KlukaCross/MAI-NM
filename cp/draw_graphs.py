import matplotlib.pyplot as plt


def plot_graphs(points1, points2):
    """
    Функция рисует два графика на основе переданных наборов точек.

    :param points1: список кортежей (x, y) для первого графика
    :param points2: список кортежей (x, y) для второго графика
    """
    # Распаковываем списки точек в отдельные списки x и y
    x1, y1 = zip(*points1)
    x2, y2 = zip(*points2)

    # Создаем фигуру и оси
    plt.figure()

    # Построение первого графика
    plt.plot(x1, y1, label='Параллельно')

    # Построение второго графика
    plt.plot(x2, y2, label='Последовательно')

    # Добавляем заголовок и подписи осей
    plt.title('Время выполнения параллельного и последовательного LUP-разложения')
    plt.xlabel('n')
    plt.ylabel('time (seconds)')

    # Отображаем легенду
    plt.legend()

    # Показать график
    plt.show()


n = int(input())
points1 = []
for i in range(n):
    points1.append(tuple(map(float, input().split())))
m = int(input())
points2 = []
for i in range(m):
    points2.append(tuple(map(float, input().split())))

plot_graphs(points1, points2)
