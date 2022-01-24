from scenario import Scenario, CyclicScenario

class Model:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def _decomposition(self, mode: str):
        pass

    def contextual_fraction(self) -> float:
        pass

    def signalling_fraction(self) -> float:
        pass

    def CbD_measure(self) -> float:
        pass

    def mix(self, other: Model, weight: float) -> Model:
        pass
