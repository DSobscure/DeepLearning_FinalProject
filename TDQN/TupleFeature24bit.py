import numpy as np

class TupleFeature24bit(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 24)
    #24bits as a block
	#AB
	#CD
    def GetIndex(self, rawBoard):
        if self.index == 1:#A
            return (rawBoard >> 72) & 0xFFFFFF
        elif self.index == 2:#B
            return (rawBoard >> 48) & 0xFFFFFF
        elif self.index == 3:#C
            return (rawBoard >> 24) & 0xFFFFFF
        elif self.index == 4:#C
            return (rawBoard >> 0) & 0xFFFFFF
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

