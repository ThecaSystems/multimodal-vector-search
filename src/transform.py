import sys
import numpy as np
import inspect
from scipy import stats
from abc import ABC, abstractmethod
from typing import Any, Type


class Transform(ABC):
    """
    This is the base class for numerical data transformations.
    The purpose of a transformation is to reduce skewness in the data and make its distribution closer to normal.
    """

    def __init__(self, data: Any) -> None:
        self.data = self.transform(data)

    @abstractmethod
    def transform(self, data: Any) -> Any:
        pass

    def non_negative_data(func):
        def wrapper(*args, **kwargs):
            data = args[1]
            if np.any(data < 0):
                raise ValueError("Input data must be non-negative.")
            return func(*args, **kwargs)

        return wrapper


class YeoJohnsonTransform(Transform):
    def __init__(self, data: Any, maxlog: float = None) -> None:
        self.maxlog = maxlog
        super().__init__(data)

    def transform(self, data: Any) -> Any:
        if isinstance(data, (float, int)):
            assert (
                self.maxlog is not None
            ), "maxlog is required for transforming scalar values"
            return stats.yeojohnson(np.array([data]), self.maxlog)[0]
        else:
            data_trans, self.maxlog = stats.yeojohnson(data)
            return data_trans


class InverseTransform(Transform):
    def __init__(self, data: Any, epsilon: float = 1e-10) -> None:
        self.epsilon = epsilon
        super().__init__(data)

    def transform(self, data: Any) -> Any:
        return 1 / (data + self.epsilon)


class LogTransform(Transform):
    def __init__(self, data: Any) -> None:
        super().__init__(data)

    @Transform.non_negative_data
    def transform(self, data: Any) -> Any:
        return np.log2(data + 1)


class SqrtTransform(Transform):
    def __init__(self, data: Any) -> None:
        super().__init__(data)

    @Transform.non_negative_data
    def transform(self, data: Any) -> Any:
        return np.sqrt(data)


def get_all_transforms() -> list[Type[Transform]]:
    current_module = inspect.currentframe().f_globals["__name__"]
    return [
        transform_class
        for name, transform_class in inspect.getmembers(sys.modules[current_module])
        if inspect.isclass(transform_class)
        and not inspect.isabstract(transform_class)
        and transform_class.__module__ == current_module
    ]


def get_best_transform(column: str) -> Type[Transform] | None:
    best_statistic = 1.0  # normality score (the lower, the better)
    best_transform = None
    for transform_class in get_all_transforms():
        try:
            transform = transform_class(column)
        except ValueError:
            continue
        test_statistic = stats.kstest(
            transform.data,
            "norm",
            args=(np.mean(transform.data), np.std(transform.data)),
        )[0]
        if test_statistic < best_statistic:
            best_statistic = test_statistic
            best_transform = transform
    return best_transform
