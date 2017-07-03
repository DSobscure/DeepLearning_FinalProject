import numpy as np

class TupleFeature(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 16)
    #4bits as a block
	#ABCD
    #EFGH
    def GetIndex(self, rawBoard):
        if self.index == 1:#ABCD
            return (rawBoard >> 16) & 0xFFFF
        elif self.index == 2:#EFGH
            return (rawBoard >> 0) & 0xFFFF
        elif self.index == 3:#ABEF
            return ((rawBoard >> 16) & 0xFF00) | ((rawBoard >> 8) & 0xFF)
        elif self.index == 4:#BCFG
            return ((rawBoard >> 12) & 0xFF00) | ((rawBoard >> 4) & 0xFF)
        elif self.index == 5:#CDGH
            return ((rawBoard >> 8) & 0xFF00) | ((rawBoard >> 0) & 0xFF)
        elif self.index == 6:#AECG
            return ((rawBoard >> 16) & 0xF0F0) | ((rawBoard >> 12) & 0xF00) | ((rawBoard >> 4) & 0xF)
        elif self.index == 7:#AEDH
            return ((rawBoard >> 16) & 0xF000) | ((rawBoard >> 12) & 0xFF0) | ((rawBoard >> 0) & 0xF)
        elif self.index == 8:#BFDH
            return ((rawBoard >> 12) & 0xF0F0) | ((rawBoard >> 0) & 0xF0F)
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

