import numpy as np

class TupleFeature16bit_FlappyBird(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 16)
    #8bits as a block total 24bit
	#A
    #B
    #C
    def GetIndex(self, rawBoard):
        if self.index == 1:#AB
            return (rawBoard >> 8) & 0xFFFF
        elif self.index == 2:#BC
            return (rawBoard >> 0) & 0xFFFF
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]
