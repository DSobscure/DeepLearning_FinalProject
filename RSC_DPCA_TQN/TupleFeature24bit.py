import numpy as np

class TupleFeature24bit(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 24)
    #8bits as a block total 32bit
	#AB
    #CD
    def GetIndex(self, rawBoard):
        if self.index == 1:#ABC
            return (rawBoard >> 8) & 0xFFFFFF
        elif self.index == 2:#ABD
            return ((rawBoard >> 8) & 0xFFFF00) | (rawBoard & 0xFF)
        elif self.index == 3:#BCD
            return rawBoard & 0xFFFFFF
        elif self.index == 4:#ACD
            return ((rawBoard >> 8) & 0xFF0000) | (rawBoard & 0xFFFF)
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]
