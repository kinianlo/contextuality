import pytest
from pytest import approx
from contextuality.model import Scenario, CyclicScenario, random_pr_like_model

def test_CbD_direction_influence():
    for _ in range(10):
        model = random_pr_like_model(n=3)
        dist = model._distributions
        assert approx(model.CbD_direct_influence()) == (2 * (abs(dist[0][0] - dist[1][0]) + 
            abs(dist[1][0] - dist[2][1]) + 
            abs(dist[2][1] + dist[0][0] - 1)))

def test_CbD_measure():
    for _ in range(10):
        model = random_pr_like_model(n=3)
        dist = model._distributions
        assert approx(model.CbD_measure()) == 2 - (2 * (abs(dist[0][0] - dist[1][0]) + 
            abs(dist[1][0] - dist[2][1]) + 
            abs(dist[2][1] + dist[0][0] - 1)))
