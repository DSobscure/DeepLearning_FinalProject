import numpy as np

class TupleFeature(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 16)

    def GetIndex(self, rawBoard):
        if self.index == 1:#ABCD
            return rawBoard & 0xFFFF
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

