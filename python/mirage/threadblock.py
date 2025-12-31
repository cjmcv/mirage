from .core import *

class TBGraph:
    def __init__(self, graph):
        self.cygraph = graph

    def new_input(
        self,
        dtensor: DTensor,
        input_map: tuple,
        store_in_dmem: bool = False,
    ):
        return self.cygraph.new_input(dtensor, input_map, store_in_dmem)