import numpy
import itertools
import picos
from scenario import Scenario, CyclicScenario, chsh_scenario, cyclic_scenario

class Model:
    def __init__(self, scenario: Scenario, distributions):
        if len(distributions) != len(scenario.contexts):
            raise ValueError('Number of distributions must be equal to the number of contexts')

        self.scenario = scenario
        self._distributions = []
        self._vector = None

        for i in range(len(distributions)):
            dist = distributions[i]
            context = scenario.contexts[i]

            if abs(sum(dist) -1) > 1e-10:
                raise ValueError(f"The distribution provided for context {context} does not sum to one")
            if len(dist) != scenario.num_outcome*len(context):
                raise ValueError(f"The distribution provided for context {context} has the wrong number of probabilities")

            self._distributions.append(numpy.array(dist))

    @property
    def vector(self) -> numpy.ndarray:
        if self._vector:
            return self._vector
        else:
            return numpy.array(self._distributions).flatten()

    def _decomposition(self, mode: str) -> float:
        M = self.scenario.incidence_matrix
        v = self.vector
        num_global_assign = M.shape[1]
        one = numpy.ones(num_global_assign)

        problem = picos.Problem()
        problem.options.solver = "mosek"

        b = picos.RealVariable('b', num_global_assign)

        problem.set_objective('max', one|b)

        problem.add_constraint(M*b <= v)

        if mode == 'cf':
            problem.add_constraint(b >= 0)
        if mode == 'sf':
            problem.add_constraint(b >= -1)
        if mode == 'sf_gz':
            problem.add_constraint(M*b >= 0)

        solution = problem.solve()
        weight_N = solution.value
        return weight_N

    def contextual_fraction(self) -> float:
        weight_NC = self._decomposition('cf')
        weight_SC = 1 - weight_NC
        return weight_SC

    def signalling_fraction(self) -> float:
        weight_NS = self._decomposition('sf')
        weight_SS = 1- weight_NS
        return weight_SS

    def CbD_measure(self) -> float:
        # Currently only for cyclic scenario with binary outcome 
        assert isinstance(self.scenario, CyclicScenario)
        assert self.scenario.num_outcome == 2

        contexts = self.scenario.contexts
        dist = self._distributions

        corr = [] 
        parity = numpy.array([1, -1, -1, 1])
        for i in range(len(contexts)):
            corr.append(sum(dist[i] * parity))

        corr_neg = list(filter(lambda x: x < 0, corr))
        s_odd = sum(numpy.abs(corr))
        if len(corr_neg) == 0:
            s_odd -= 2 * min(corr)
        elif len(corr_neg) % 2 == 0:
            s_odd += 2 * max(corr_neg)

        ev_in = []
        ev_out = []
        parity_in = numpy.array([1, 1, -1, -1])
        parity_out = numpy.array([1, -1, 1, -1])
        for i in range(len(contexts)):
            ev_in.append(sum(dist[i] * parity_in))
            ev_out.append(sum(dist[i] * parity_out))
        Delta = sum(numpy.abs(ev_in - numpy.roll(ev_out, 1)))

        return s_odd - Delta - (len(contexts) - 2)

    def mix(self, other, weight: float):
        if self.scenario != other.scenario:
            raise ValueError("Cannot mix two models with non-identical scenarios")
        
        self_dist = self._distributions
        other_dist = other._distributions

        mix_dist = []
        for i in range(len(self_dist)):
            mix_dist.append((1-weight) * self_dist[i] + weight * other_dist[i])

        mix_model = Model(self.scenario, mix_dist)
        return mix_model

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

def pr_model(n: int=None) -> Model:
    if not n:
        table = [[1/2, 0, 0, 1/2],
                [1/2, 0, 0, 1/2],
                [1/2, 0, 0, 1/2],
                [0, 1/2, 1/2, 0]]
        return Model(chsh_scenario(), table)
    else:
        table = numpy.zeros((n,4))
        table[:-1, 0] = 1/2
        table[:-1, -1] = 1/2
        table[-1, 1] = 1/2
        table[-1, 2] = 1/2
        return Model(cyclic_scenario(n), table)


def bell_model() -> Model:
    scenario = chsh_scenario()
    table = [[4/8, 0/8, 0/8, 4/8],
             [3/8, 1/8, 1/8, 3/8],
             [3/8, 1/8, 1/8, 3/8],
             [1/8, 3/8, 3/8, 1/8]]
    return Model(chsh_scenario(), table)

def random_model(scenario: Scenario):
    contexts = scenario.contexts
    num_outcome = scenario.num_outcome
    table = []
    for context in scenario.contexts:
        rand_dist = numpy.random.rand(num_outcome**len(context))
        rand_dist = rand_dist/sum(rand_dist)
        table.append(rand_dist)
    return Model(scenario, table)


if __name__ == "__main__":
    s = CyclicScenario(['x1', 'x2', 'x3'], 2)
    dist = [[1/2, 0, 0, 1/2], [1/2, 0, 0, 1/2], [0, 1/2, 1/2, 0]]
    dist2 = [[1/2, 0, 0, 1/2], [1/2, 0, 0, 1/2], [1/2, 0, 0, 1/2]]
    m = Model(s, dist)
    m2 = Model(s, dist2)
    m = m.mix(m2, 0.9)
    print(str(m))
    print(f'CF:\t{m.contextual_fraction():.5f}')
    print(f'SF:\t{m.signalling_fraction():.5f}')
    print(f'CbD:\t{m.CbD_measure():.5f}')
