from random import random


def test_print_model_cbd():
    from contextuality import chsh_scenario, random_model, print_model_cbd
    scenario = chsh_scenario()
    for _ in range(10):
        model = random_model(scenario)
        print_model_cbd(model)

def test_symmetrized_model():
    from contextuality import symmetrized_model
    from contextuality import pr_model, random_pr_like_model
    for n in range(3, 10):
        assert symmetrized_model(pr_model(n)) == pr_model(n)
        assert symmetrized_model(random_pr_like_model(n)) == pr_model(n)

    