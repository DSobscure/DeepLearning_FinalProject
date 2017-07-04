import numpy as np

class TupleFeature(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 24)

    def GetIndex(self, rawBoard):
        #4 bit as a block total 8 block
        #ABCD
        #EFGH
        if self.index == 1:#ABCDEF
            return (rawBoard >> 8) & 0xFFFFFF
        elif self.index == 2:#BCDEFG
            return (rawBoard >> 4) & 0xFFFFFF
        elif self.index == 3:#CDEFGH
            return (rawBoard >> 0) & 0xFFFFFF
        elif self.index == 4:#ABCEFG
            return ((rawBoard >> 8) & 0xFFF000) | ((rawBoard >> 4) & 0xFFF)
        elif self.index == 5:#BCDFGH
            return ((rawBoard >> 4) & 0xFFF000) | ((rawBoard >> 0) & 0xFFF)
        else:
            assert False
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

