import numpy
import itertools

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


class CyclicScenario(Scenario):
    def __init__(self, observables: list[str], num_outcome: int):
        self._observables = observables
        contexts = []
        for i in range(len(observables)):
            i_next = (i + 1) % len(observables)
            contexts.append((observables[i], observables[i_next]))

        super().__init__(contexts, num_outcome)

def chsh_scenario() -> CyclicScenario:
    observables = ['a1', 'b1', 'a2', 'b2']
    return CyclicScenario(observables, 2)

def cyclic_scenario(n: int = 3):
    observables = [f'x{i+1}' for i in range(n)]
    return CyclicScenario(observables, 2)

if __name__ == "__main__":
    s = CyclicScenario(['x1', 'x2', 'x3'], 2)
    print(repr(s.incidence_matrix))
    print(s.__repr__())
