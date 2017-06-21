import numpy as np

class TupleFeature24bit(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 24)
    #24bits as a block
	#ABCDEF
    def GetIndex(self, rawBoard):
        if self.index == 1:#A
            return (rawBoard >> 120) & 0xFFFFFF
        elif self.index == 2:#B
            return (rawBoard >> 96) & 0xFFFFFF
        elif self.index == 3:#C
            return (rawBoard >> 72) & 0xFFFFFF
        elif self.index == 4:#D
            return (rawBoard >> 48) & 0xFFFFFF
        elif self.index == 5:#E
            return (rawBoard >> 24) & 0xFFFFFF
        elif self.index == 6:#F
            return (rawBoard >> 0) & 0xFFFFFF
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

