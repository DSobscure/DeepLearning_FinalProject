import numpy as np

class TupleFeature24bit(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 24)
    #4bits as a block
	#ABCDEF
	#GHIJKL
    #MNOPQR
    def GetIndex(self, rawBoard):
        if self.index == 1:#ABCDEF
            return (rawBoard >> 48) & 0xFFFFFF
        elif self.index == 2:#GHIJKL
            return (rawBoard >> 24) & 0xFFFFFF
        elif self.index == 3:#MNOPQR
            return (rawBoard) & 0xFFFFFF
        elif self.index == 4:#ABCGHI
            return ((rawBoard >> 48) & 0xFFF000) | ((rawBoard >> 36) & 0xFFF)
        elif self.index == 5:#DEFJKL
            return ((rawBoard >> 36) & 0xFFF000) | ((rawBoard >> 24) & 0xFFF)
        elif self.index == 6:#GHIMNO
            return ((rawBoard >> 24) & 0xFFF000) | ((rawBoard >> 12) & 0xFFF)
        elif self.index == 7:#DEFJKL
            return ((rawBoard >> 12) & 0xFFF000) | ((rawBoard) & 0xFFF)
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

