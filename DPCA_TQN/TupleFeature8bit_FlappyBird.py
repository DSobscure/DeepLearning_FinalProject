import numpy as np

class TupleFeature8bit_FlappyBird(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 8)
    #8bits as a block total 24bit
	#A
    #B
    #C
    def GetIndex(self, rawBoard):
        if self.index == 1:#A
            return (rawBoard >> 16) & 0xFF
        elif self.index == 2:#B
            return (rawBoard >> 8) & 0xFF
        elif self.index == 3:#C
            return (rawBoard >> 0) & 0xFF
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

