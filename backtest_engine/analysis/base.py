from __future__ import annotations

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class AnalysisReport(ABC):
    """
    Shared interface for the analysis layer.

    Derived analytics classes compute eagerly inside __init__. report() always plots and
    prints self, then returns self.
    """

    def report(self) -> None:
        """
        Display the report: plot figures and print metric table.
        """

        self.plot()
        print(self)

    @abstractmethod
    def plot(self) -> list[plt.Figure]:
        """
        Build, show, and return this report's figures.
        """

        return []

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a table of derived analytics.
        """

        return ""
