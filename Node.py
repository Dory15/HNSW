import numpy as np

class Node:
    def __init__(self, id, vector):
        self.id = id
        self.vector = vector
        self.neighboards_per_level = {}