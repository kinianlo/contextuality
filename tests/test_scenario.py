import pytest
import numpy as np
from contextuality.model import Scenario, CyclicScenario

def test_incidence_matrix():
    contexts = [('x1', 'x2'), ('x2', 'x3'), ('x3', 'x1')]
    num_outcome = 2
    s = Scenario(contexts, num_outcome)
    expected = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1],
                         [1, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0, 1],
                         [1, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 1]])
    assert np.all(s.incidence_matrix == expected)

def test_observables():
    contexts = [('x1', 'x2'), ('x2', 'x3'), ('x3', 'x1')]
    num_outcome = 2
    s = Scenario(contexts, num_outcome)

    expected = ['x1', 'x2', 'x3']
    assert s.observables == expected

def test_cyclic_scenario():
    observables = ['x1', 'x2', 'x3']
    expected = [('x1', 'x2'), ('x2', 'x3'), ('x3', 'x1')]
    num_outcome = 2

    s = CyclicScenario(observables, num_outcome)
    assert s.contexts == expected

def test_scenario_observables():

