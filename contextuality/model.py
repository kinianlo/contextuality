import numpy
import itertools
from scenario import Scenario, CyclicScenario

class Model:
    def __init__(self, scenario: Scenario, distributions):
        if len(distributions) != len(scenario.contexts):
            raise ValueError('Number of distributions must be equal to the number of contexts')

        self.scenario = scenario
        self._distributions = []

        for i in range(len(distributions)):
            dist = distributions[i]
            context = scenario.contexts[i]

            if abs(sum(dist) -1) > 1e-10:
                raise ValueError(f"The distribution provided for context {context} does not sum to one")
            if len(dist) != scenario.num_outcome*len(context):
                raise ValueError(f"The distribution provided for context {context} has the wrong number of probabilities")

            self._distributions.append(numpy.array(dist))

    def _decomposition(self, mode: str):
        pass

    def contextual_fraction(self) -> float:
        pass

    def signalling_fraction(self) -> float:
        pass

    def CbD_measure(self) -> float:
        pass

    def mix(self, other, weight: float):
        pass

    def __str__(self):
        num_outcome = self.scenario.num_outcome;
        contexts = self.scenario.contexts
        out = ""
        if isinstance(self.scenario, CyclicScenario):
            context_strs = [str(c).replace("'", "")+' ' for c in contexts]
            context_col_width = max(map(len, context_strs))
            assigns = map(str, itertools.product(range(2), repeat=num_outcome))
            assigns_str = ' '.join(assigns)
            out += ' '*context_col_width + assigns_str + '\n'
            for i, c in enumerate(contexts):
                dist_str = ' '.join([f"{p:>6.4f}" for p in self._distributions[i]])
                out += context_strs[i] + dist_str + '\n'
        else:
            for i, c in enumerate(contexts):
                assigns = map(str, itertools.product(range(len(c)), repeat=num_outcome))
                assigns_str = ' '.join(assigns)
                context_str = str(c).replace("'", "") + "->"
                dist_str = ' '.join([f"{p:>6.4f}" for p in self._distributions[i]])

                out += context_str + assigns_str + '\n'
                out += ' '*len(context_str) + dist_str + '\n'
        return out

if __name__ == "__main__":
    s = CyclicScenario(['x1', 'x2', 'x3'], 2)
    dist = [[1/2, 0, 0, 1/2], [1/2, 0, 0, 1/2], [0, 1/2, 1/2, 0]]
    m = Model(s, dist)
    print(str(m))
