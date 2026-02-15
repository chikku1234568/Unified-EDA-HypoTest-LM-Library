"""
Registry for statistical tests in the hypotest framework.

This module provides a central registry (`TestRegistry`) that tracks all
concrete statistical test classes, along with a decorator (`register_test`)
that simplifies registration.

The registry enables dynamic discovery of available tests, filtering by
applicability to a given dataset, and instantiation of tests by name.

Examples
--------
>>> from hypotest.tests.base import StatisticalTest, TestMetadata
>>> from hypotest.core.dataset import Dataset
>>> 
>>> @register_test
>>> class IndependentTTest(StatisticalTest):
...     metadata = TestMetadata(
...         name="Independent t‑test",
...         category="parametric",
...         assumptions=["normality", "variance_homogeneity"],
...         applicable_types={"target": ["continuous"], "feature": ["binary"]},
...         min_sample_size=20,
...         description="Compares means between two independent groups."
...     )
...     # implement abstract methods...
...     pass
>>> 
>>> test = TestRegistry.get_test("Independent t‑test")
>>> isinstance(test, IndependentTTest)
True
>>> 
>>> dataset = Dataset(...)
>>> applicable = TestRegistry.find_applicable_tests(dataset, "score", ["group"])
>>> len(applicable) >= 1
True
"""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from .base import StatisticalTest
from ..core.dataset import Dataset

_logger = logging.getLogger(__name__)


class TestRegistry:
    """
    Central registry for statistical test classes.

    The registry maintains a mapping from test name (as defined in the test's
    metadata) to the test class. Registration can be performed via the class
    method `register` or via the `register_test` decorator.

    The registry is class‑level; all instances share the same mapping.

    Attributes
    ----------
    _registry : Dict[str, Type[StatisticalTest]]
        Mapping from test name to test class. This attribute is private and
        should only be accessed through the class methods.

    Examples
    --------
    >>> TestRegistry.register(IndependentTTest)
    <class 'IndependentTTest'>
    >>> TestRegistry.get_test("Independent t‑test")
    IndependentTTest(name='Independent t‑test')
    """

    _registry: Dict[str, Type[StatisticalTest]] = {}

    @classmethod
    def register(cls, test_class: Type[StatisticalTest]) -> Type[StatisticalTest]:
        """
        Register a statistical test class.

        The test is registered under its canonical name
        (`test_class.metadata.name`). If a test with the same name is already
        registered, a `ValueError` is raised.

        Parameters
        ----------
        test_class : Type[StatisticalTest]
            A concrete subclass of `StatisticalTest` that defines a `metadata`
            attribute of type `TestMetadata`.

        Returns
        -------
        Type[StatisticalTest]
            The same test class, allowing the method to be used as a decorator.

        Raises
        ------
        ValueError
            If `test_class` does not have a `metadata` attribute, or if the
            metadata's `name` is already registered.
        TypeError
            If `test_class` is not a subclass of `StatisticalTest`.

        Examples
        --------
        >>> TestRegistry.register(IndependentTTest)
        <class 'IndependentTTest'>
        """
        if not isinstance(test_class, type) or not issubclass(test_class, StatisticalTest):
            raise TypeError(
                f"Test class must be a subclass of StatisticalTest, got {test_class}"
            )

        try:
            name = test_class.metadata.name
        except AttributeError as e:
            raise ValueError(
                f"Test class {test_class.__name__} must define a `metadata` attribute."
            ) from e

        if name in cls._registry:
            raise ValueError(
                f"A test with name '{name}' is already registered. "
                f"Registered test: {cls._registry[name]}"
            )

        cls._registry[name] = test_class
        _logger.debug("Registered test '%s' (%s)", name, test_class.__name__)
        return test_class

    @classmethod
    def get_test(cls, name: str) -> StatisticalTest:
        """
        Retrieve an instance of a registered test by name.

        Parameters
        ----------
        name : str
            Canonical name of the test (as defined in its metadata).

        Returns
        -------
        StatisticalTest
            An instance of the registered test class.

        Raises
        ------
        ValueError
            If no test with the given name is registered.

        Examples
        --------
        >>> test = TestRegistry.get_test("Independent t‑test")
        >>> test.metadata.name
        'Independent t‑test'
        """
        if name not in cls._registry:
            raise ValueError(
                f"No test named '{name}' is registered. "
                f"Available tests: {list(cls._registry.keys())}"
            )
        test_class = cls._registry[name]
        try:
            return test_class()
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate test '{name}' ({test_class.__name__}): {e}"
            ) from e

    @classmethod
    def get_all_tests(cls) -> List[StatisticalTest]:
        """
        Return a list of instantiated registered tests.

        The order of the list is the order of registration (insertion order
        of Python 3.7+ dictionaries).

        Returns
        -------
        List[StatisticalTest]
            List of test instances.

        Examples
        --------
        >>> tests = TestRegistry.get_all_tests()
        >>> len(tests) == len(TestRegistry._registry)
        True
        """
        tests = []
        for name, test_class in cls._registry.items():
            try:
                tests.append(test_class())
            except Exception as e:
                _logger.warning(
                    "Could not instantiate test '%s' (%s): %s",
                    name, test_class.__name__, e
                )
        return tests

    @classmethod
    def find_applicable_tests(
        cls,
        dataset: Dataset,
        target: str,
        features: List[str],
    ) -> List[StatisticalTest]:
        """
        Return all registered tests that are applicable to the given dataset.

        A test is considered applicable if its `is_applicable` method returns
        `True` for the provided dataset, target, and features.

        Parameters
        ----------
        dataset : Dataset
            Dataset container with the underlying data.
        target : str
            Name of the target (dependent) variable.
        features : List[str]
            Names of the feature (independent) variables.

        Returns
        -------
        List[StatisticalTest]
            List of instantiated tests that are applicable, in the same order
            as registration.

        Examples
        --------
        >>> dataset = Dataset(...)
        >>> applicable = TestRegistry.find_applicable_tests(dataset, "score", ["group"])
        >>> [t.metadata.name for t in applicable]
        ['Independent t‑test', 'Mann‑Whitney U']
        """
        applicable = []
        for name, test_class in cls._registry.items():
            try:
                test_instance = test_class()
                if test_instance.is_applicable(dataset, target, features):
                    applicable.append(test_instance)
            except Exception as e:
                _logger.debug(
                    "Test '%s' raised an exception during applicability check: %s",
                    name, e
                )
                continue
        return applicable

    @classmethod
    def clear(cls) -> None:
        """
        Clear the registry (primarily for testing).

        This method removes all registered tests, restoring the registry to an
        empty state. It should not be used in production except for isolated
        test environments.

        Examples
        --------
        >>> TestRegistry.clear()
        >>> len(TestRegistry._registry)
        0
        """
        cls._registry.clear()
        _logger.debug("Registry cleared")

    @classmethod
    def get_registered_names(cls) -> List[str]:
        """
        Return a list of the names of all registered tests.

        Returns
        -------
        List[str]
            List of test names, in registration order.

        Examples
        --------
        >>> TestRegistry.get_registered_names()
        ['Independent t‑test', 'Pearson correlation']
        """
        return list(cls._registry.keys())


def register_test(test_class: Type[StatisticalTest]) -> Type[StatisticalTest]:
    """
    Decorator that registers a statistical test class with the TestRegistry.

    This is a convenience function that calls `TestRegistry.register(test_class)`.
    It allows test classes to be registered using a simple decorator syntax.

    Parameters
    ----------
    test_class : Type[StatisticalTest]
        The test class to register.

    Returns
    -------
    Type[StatisticalTest]
        The same test class, unchanged.

    Examples
    --------
    >>> @register_test
    ... class IndependentTTest(StatisticalTest):
    ...     metadata = TestMetadata(...)
    ...     # implement abstract methods...
    ...     pass
    ...
    >>> TestRegistry.get_registered_names()
    ['Independent t‑test']
    """
    TestRegistry.register(test_class)
    return test_class