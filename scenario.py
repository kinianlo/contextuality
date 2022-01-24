class Scenario:
    def __init__(self, contexts: list[tuple[str, ...]], num_outcome: int):
        self.contexts = contexts
        self.num_outcome = num_outcome

    @property
    def observables(self) -> list[str]:
        out = []
        for context in self.contexts:
            for observable in context:
                if observable not in out:
                    out.append(observable)
        return out

class CyclicScenario(Scenario):
    def __init__(self, observables: list[str], num_outcome: int):
        contexts = []
        for i in range(len(observables)):
            i_next = (i + 1) % len(observables)
            contexts.append((observables[i], observables[i_next]))

        super().__init__(contexts, num_outcome)


