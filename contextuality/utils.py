from typing import TYPE_CHECKING


def sum_odd(arr):
    """
    Compute the maximum value of the sum of the given 
    array `arr` with odd number of elements first mutliplied
    by -1.
    """
    arr_neg = [a for a in arr if a <= 0]
    arr_abs = [abs(a) for a in arr]

    s = sum(arr_abs)

    if len(arr_neg) % 2 == 0:
        return s - 2 * min(arr_abs)
    else:
        return s

if TYPE_CHECKING:
    from contextuality.model import Model

def print_model_cbd(model: 'Model'):
    from prettytable import PrettyTable
    from contextuality import Model, CyclicScenario
    outcomes = model.scenario.outcomes
    assert isinstance(model.scenario, CyclicScenario)
    for i, context in enumerate(model.scenario.contexts):
        pt = PrettyTable()
        pt.field_names = [context] + outcomes
        for o1, outcome1 in enumerate(outcomes):
            row = []
            row.append(outcome1)
            for o2, outcome2 in enumerate(outcomes):
                row.append(f"{model.probability(context, (o1, o2)):.4f}")
            pt.add_row(row)
        print(pt)
                
        