import numpy
import itertools
import picos

class Scenario:
    def __init__(self, contexts: list[tuple[str, ...]], num_outcome: int):
        self.contexts = contexts
        self.num_outcome = num_outcome

        self._observables = None
        self._incidence_matrix = None

    @property
    def observables(self) -> list[str]:
        if self._observables:
            return self._observables

        out = []
        for context in self.contexts:
            for observable in context:
                if observable not in out:
                    out.append(observable)
        return out

    @property
    def incidence_matrix(self) -> numpy.ndarray:
        if self._incidence_matrix:
            return self._incidence_matrix
        global_assigns = list(itertools.product(range(self.num_outcome), repeat=len(self.observables)))

        M_context = []
        for context in self.contexts:
            local_assigns = list(itertools.product(range(self.num_outcome), repeat=len(context)))

            ob_idx = [self.observables.index(ob) for ob in context]
            g_context = [tuple([g[idx] for idx in ob_idx]) for g in global_assigns]

            M_context_this = numpy.zeros((len(local_assigns), len(global_assigns)), dtype=int)
            for i in range(len(local_assigns)):
                for j in range(len(g_context)):
                    M_context_this[i, j] = local_assigns[i] == g_context[j]

            M_context.append(M_context_this)

        M = numpy.vstack(M_context)
        return M

    def __eq__(self, other):
        return self.contexts == other.contexts and self.num_outcome == other.num_outcome

    def __repr__(self):
        class_name = self.__class__.__name__
        contexts_repr = self.contexts.__repr__()
        return f"{class_name}({contexts_repr}, {self.num_outcome})"

    def __str__(self):
        observable_str = ', '.join([ob for ob in self.observables])
        context_str = ', '.join([str(c).replace("'", "") for c in self.contexts])
        outcome_str = ', '.join(map(str, range(self.num_outcome)))
        out = ''
        out += f"Observables\t[{observable_str}]\n"
        out += f"Contexts\t[{context_str}]\n"
        out += f"Outcomes\t[{outcome_str}]"
        return out


class CyclicScenario(Scenario):
    def __init__(self, observables: list[str], num_outcome: int):
        self._observables = observables
        contexts = []
        for i in range(len(observables)):
            i_next = (i + 1) % len(observables)
            contexts.append((observables[i], observables[i_next]))

        super().__init__(contexts, num_outcome)

    def __repr__(self):
        class_name = self.__class__.__name__
        observables_repr = self.observables.__repr__()
        return f"{class_name}({observables_repr}, {self.num_outcome})"
        

class Model:
    def __init__(self, scenario: Scenario, distributions):
        if len(distributions) != len(scenario.contexts):
            raise ValueError('Number of distributions must be equal to the number of contexts')

        self.scenario = scenario
        self._distributions = []
        self._vector = None

        if isinstance(scenario, CyclicScenario):
            self._distributions = numpy.array(distributions)
            num_outcome = scenario.num_outcome
            rank = len(scenario.contexts)
            expected_shape = (rank, num_outcome**2)
            if self._distributions.shape != expected_shape:
                raise ValueError("The distributions provided are not compatible with the scenario.")
        else:
            for i in range(len(distributions)):
                context = scenario.contexts[i]
                self._distributions.append(numpy.array(distributions[i]))

                if len(dist) != scenario.num_outcome*len(context):
                    raise ValueError(f"The distribution provided for context {context} has the wrong number of probabilities")

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

    def __distributions_binary_operation(self, other, operation):
        if self.scenario is not other.scenario and self.scenario != other.scenario:
            raise ValueError("Incompatible scenarios.")
        is_self_cyclic = isinstance(self.scenario, CyclicScenario)
        is_other_cyclic = isinstance(other.scenario, CyclicScenario)
        if is_self_cyclic or is_other_cyclic:
            dist = operation(self._distributions, other._distributions)
            if is_self_cyclic:
                return Model(self.scenario, dist) 
            else:
                return Model(other.scenario, dist)
        else:
            dist = []
            self_dist = self.scenario._distributions
            other_dist = other.scenario._distributions
            for i in range(len(self.scenario.contexts)):
                dist.append(operation(self_dist[i], self_dist[i]))
            return Model(self.scenario, dist)

    def __add__(self, other):
        return self.__distributions_binary_operation(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self.__distributions_binary_operation(other, lambda x, y: x - y)

    def __mul__(self, scaler):
        if isinstance(self.scenario, CyclicScenario):
            dist = self._distributions * scaler
        else:
            dist = [d * scaler for d in self._distributions]
        return Model(self.scenario, self._distributions * scaler)

    def __rmul__(self, scaler):
        return self.__mul__(scaler)

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

    def __eq__(self, other):
        if self.scenario != other.scenario:
            return False
        if isinstance(self.scenario, CyclicScenario) or isinstance(other.scenario, CyclicScenario):
            return numpy.allclose(self._distributions, other._distributions)

    def __repr__(self):
        class_name = self.__class__.__name__
        scenario_repr = self.scenario.__repr__()
        dist_repr = self._distributions.__repr__()
        return f"{class_name}({scenario_repr}, {dist_repr})"


def chsh_scenario() -> CyclicScenario:
    observables = ['a1', 'b1', 'a2', 'b2']
    return CyclicScenario(observables, 2)

def cyclic_scenario(n: int = 3):
    observables = [f'x{i+1}' for i in range(n)]
    return CyclicScenario(observables, 2)

def get_model_from_vector(scenario, vector) -> Model:
    num_assign = [scenario.num_outcome ** len(c) for c in scenario.contexts]
    context_last_idx = numpy.add.accumulate(num_assign)
    dist = numpy.split(vector, context_last_idx[:-1])
    return Model(scenario, dist)

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

def random_pr_like_model(n: int=3):
    table = numpy.zeros((n, 4))
    table[:-1, 0] = numpy.random.rand(n-1)
    table[:-1, -1] = numpy.random.rand(n-1)
    table[-1, 1] = numpy.random.rand()
    table[-1, 2] = numpy.random.rand()
    table = table / numpy.sum(table, axis=1, keepdims=True)
    return Model(cyclic_scenario(n), table)

def bell_model() -> Model:
    scenario = chsh_scenario()
    table = [[4/8, 0/8, 0/8, 4/8],
             [3/8, 1/8, 1/8, 3/8],
             [3/8, 1/8, 1/8, 3/8],
             [1/8, 3/8, 3/8, 1/8]]
    return Model(chsh_scenario(), table)

def random_model(scenario: Scenario, scaling=None):
    contexts = scenario.contexts
    num_outcome = scenario.num_outcome
    table = []
    for i in range(len(contexts)):
        context = contexts[i]
        rand_dist = numpy.random.rand(num_outcome**len(context))
        if scaling:
            rand_dist = rand_dist * scaling[i]
        rand_dist = rand_dist/sum(rand_dist)
        table.append(rand_dist)
    return Model(scenario, table)


if __name__ == "__main__":
    s = CyclicScenario(['x1', 'x2', 'x3'], 2)
    print(s)
    dist1 = [[1/2, 0, 0, 1/2], [1/2, 0, 0, 1/2], [0, 1/2, 1/2, 0]]
    dist2 = [[1/2, 0, 0, 1/2], [1/2, 0, 0, 1/2], [1/2, 0, 0, 1/2]]
    m1 = Model(s, dist1)
    m2 = Model(s, dist2)
    # m = m1.mix(m2, 0.9)
    weight = 0.9
    m = weight * m1 + (1-weight) * m2
    print(str(m))
    print(f'CF:\t{m.contextual_fraction():.5f}')
    print(f'SF:\t{m.signalling_fraction():.5f}')
    print(f'CbD:\t{m.CbD_measure():.5f}')
    print(m.__repr__())

    print(get_model_from_vector(s, m.vector))
