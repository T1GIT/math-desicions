import unittest
from io import StringIO
from unittest import mock
from tasks import task_9


class TestTask9(unittest.TestCase):

    @mock.patch('sys.stdout', new_callable=StringIO)
    @mock.patch('builtins.input')
    def test_csv(self, mock_input, mock_stdout):
        mock_input.side_effect = ['csv']
        task_9.run()
        report = mock_stdout.getvalue()

        self.assertIn('Оптимальная чистая стратегия для игрока А: ПК', report)
        self.assertIn('Цена игры для игрока А при выборе чистой оптимальной стратегии: 240', report)
        self.assertIn('Оптимальная чистая стратегия для игрока B: Телефоны', report)
        self.assertIn('Цена игры для игрока B при выборе чистой оптимальной стратегии: 300', report)
        self.assertIn('70         0  30', report)
        self.assertIn('Цена игры для игрока А при выборе смешанной оптимальной стратегии: 282', report)
        self.assertIn('100         0         0', report)
        self.assertIn('Цена игры для игрока B при выборе смешанной оптимальной стратегии: 200', report)


    @mock.patch('sys.stdout', new_callable=StringIO)
    @mock.patch('builtins.input')
    def test_random(self, mock_input, mock_stdout):
        mock_input.side_effect = [
            'csv',
            '5',
            '10'
        ]
        task_9.run()
        report = mock_stdout.getvalue()

        self.assertIn('Оптимальная чистая стратегия для игрока А: ', report)
        self.assertIn('Цена игры для игрока А при выборе чистой оптимальной стратегии: ', report)
        self.assertIn('Оптимальная чистая стратегия для игрока B: ', report)
        self.assertIn('Цена игры для игрока B при выборе чистой оптимальной стратегии: ', report)
        self.assertIn('Цена игры для игрока А при выборе смешанной оптимальной стратегии: ', report)
        self.assertIn('Цена игры для игрока B при выборе смешанной оптимальной стратегии: ', report)

    @mock.patch('sys.stdout', new_callable=StringIO)
    @mock.patch('builtins.input')
    def test_keyboard(self, mock_input, mock_stdout):
        mock_input.side_effect = [
            'keyboard',
            'Телефоны Ноутбуки ПК',
            'Телефоны Наушники Ноутбуки',
            '300 280 230',
            '200 180 130',
            '240 380 400'
        ]
        task_9.run()
        report = mock_stdout.getvalue()

        self.assertIn('Оптимальная чистая стратегия для игрока А: ПК', report)
        self.assertIn('Цена игры для игрока А при выборе чистой оптимальной стратегии: 240', report)
        self.assertIn('Оптимальная чистая стратегия для игрока B: Телефоны', report)
        self.assertIn('Цена игры для игрока B при выборе чистой оптимальной стратегии: 300', report)
        self.assertIn('70         0  30', report)
        self.assertIn('Цена игры для игрока А при выборе смешанной оптимальной стратегии: 282', report)
        self.assertIn('100         0         0', report)
        self.assertIn('Цена игры для игрока B при выборе смешанной оптимальной стратегии: 200', report)


if __name__ == '__main__':
    unittest.main()