from .Environment import Environment
class UnimodalEnvironment(Environment):
    def __init__(self, list_of_arms, graph=None):
        super().__init__(list_of_arms)
        self.graph = graph



    def getNeighbors(arm,
                    arm_included=True):
        pass
