import numpy as np

class TupleFeature12bit(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 12)
    #4bits as a block
	#ABCD
	#EFGH
    def GetIndex(self, rawBoard):
        if self.index == 1:#ABCD
            return (((rawBoard >> 12) & 0xFFF))
        elif self.index == 2:#EFGH
            return (((rawBoard) & 0xFFF))
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

