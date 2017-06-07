import numpy as np

class TupleFeature24bit(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 24)
    #4bits as a block
	#ABCD
	#EFGH
    def GetIndex(self, rawBoard):
        if self.index == 1:#ABCEFG
            return ((rawBoard >> 8) & 0xFFF000) | ((rawBoard >> 4) & 0xFFF)
        elif self.index == 2:#BCDFGH
            return ((rawBoard >> 4) & 0xFFF000) | ((rawBoard >> 0) & 0xFFF)
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

