from abc import ABC, abstractmethod


class AbstractTask(ABC):

    @abstractmethod
    def input(self, default: bool) -> None:
        """
        Reads the data from the user's input.

        :param default: if True doesn't request the data
        """

    @abstractmethod
    def output(self, *args, **kwargs) -> None:
        """
        Writes the calculated data into the console.
        Requires self.input() and self.calc() before call.

        :param accuracy: amount of ciphers after the dot
        """

    @abstractmethod
    def calc(self) -> None:
        """
        Calculates the task.
        Requires self.input(*args) before call.
        """

    def chart(self) -> None:
        """
        Visualize the calculated task.
        Requires self.input() and self.calc() before call.
        """
