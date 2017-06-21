import numpy as np

class TupleFeature16bit(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 16)
    #16bits as a block
	#ABCD
    def GetIndex(self, rawBoard):
        if self.index == 1:#A
            return (((rawBoard >> 48) & 0xFFFF))
        elif self.index == 2:#B
            return (((rawBoard >> 32) & 0xFFFF))
        elif self.index == 3:#C
            return (((rawBoard >> 16) & 0xFFFF))
        elif self.index == 4:#D
            return (((rawBoard >> 0) & 0xFFFF))
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

