import numpy as np

class FourTupleFeature(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(65536)
    
    def GetIndex(self, rawBoard):
        if self.index == 1:#1,2,3,4
            return (((rawBoard >> 48) & 0xFFFF))
        elif self.index == 2:#5,6,7,8
            return (((rawBoard >> 32) & 0xFFFF))
        elif self.index == 3:#9, 10, 11, 12
            return (((rawBoard >> 16) & 0xFFFF))
        elif self.index == 4:#13, 14, 15, 16
            return (((rawBoard) & 0xFFFF))
        elif self.index == 5:#6,7,10,11
            return (((rawBoard >> 28) & 0xFF00) | ((rawBoard >> 20) & 0xFF))
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

