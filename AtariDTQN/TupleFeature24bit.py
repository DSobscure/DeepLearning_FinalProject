import numpy as np

class TupleFeature24bit(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 24)
    #4bits as a block
	#ABCDEF
	#GHIJKL
    def GetIndex(self, rawBoard):
        if self.index == 1:#ABCDEF
            return (rawBoard >> 24) & 0xFFFFFF
        elif self.index == 2:#GHIJKL
            return (rawBoard) & 0xFFFFFF
        elif self.index == 3:#ABCGHI
            return ((rawBoard >> 24) & 0xFFF000) | ((rawBoard >> 12) & 0xFFF)
        elif self.index == 4:#BCDHIJ
            return ((rawBoard >> 20) & 0xFFF000) | ((rawBoard >> 8) & 0xFFF)
        elif self.index == 5:#CDEIJK
            return ((rawBoard >> 16) & 0xFFF000) | ((rawBoard >> 4) & 0xFFF)
        elif self.index == 6:#DEFJKL
            return ((rawBoard >> 12) & 0xFFF000) | ((rawBoard >> 0) & 0xFFF)
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

