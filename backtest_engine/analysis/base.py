from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class AnalysisReport(ABC):
    """
    Shared interface for the analysis layer.

    Derived analytics classes compute eagerly inside __init__. report() always plots and
    prints self.
    """

    def report(self, *, savepath: Optional[str | Path] = None) -> None:
        """
        Display the report: plot figures and print metric table.

        savepath : Optional[str | Path] = None
            If given, this run's figures are saved under a new f"{ClassName}_{timestamp}"
            directory inside savepath, then closed. Figures still display either way.
        """

        self.plot(savepath=savepath)
        print(self)

    @abstractmethod
    def plot(self, *, savepath: Optional[str | Path] = None) -> None:
        """
        Build and show this report's figures.

        savepath : Optional[str | Path] = None
            If given, this run's figures are saved under a new f"{ClassName}_{timestamp}"
            directory inside savepath, then closed. Figures still display either way.
        """

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a table of derived analytics.
        """

        return ""
