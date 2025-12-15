from .core import *

class TBGraph:
    def __init__(self, graph):
        self.cygraph = graph

    def new_input(
        self,
        dtensor: DTensor,
        input_map: tuple,
        forloop_dim: int,
        store_in_dmem: bool = False,
    ):
        return self.cygraph.new_input(dtensor, input_map, forloop_dim, store_in_dmem)

    # def new_output(self, stensor: STensor, output_map: tuple, forloop_dim: int = -1):
    #     return self.cygraph.new_output(stensor, output_map, forloop_dim)