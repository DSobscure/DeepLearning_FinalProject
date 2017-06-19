import numpy as np

class TupleFeature16bit(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 16)
    #8bits as a block
	#AB
	#CD
    #EF
    #GH
    def GetIndex(self, rawBoard):
        if self.index == 1:#AB
            return (((rawBoard >> 48) & 0xFFFF))
        elif self.index == 2:#CD
            return (((rawBoard >> 32) & 0xFFFF))
        elif self.index == 3:#EF
            return (((rawBoard >> 16) & 0xFFFF))
        elif self.index == 4:#GH
            return (((rawBoard >> 0) & 0xFFFF))
        elif self.index == 5:#AC
            return (((rawBoard >> 48) & 0xFF00) | ((rawBoard >> 40) & 0xFF))
        elif self.index == 6:#BD
            return (((rawBoard >> 40) & 0xFF00) | ((rawBoard >> 32) & 0xFF))
        elif self.index == 7:#CE
            return (((rawBoard >> 32) & 0xFF00) | ((rawBoard >> 24) & 0xFF))
        elif self.index == 8:#DF
            return (((rawBoard >> 24) & 0xFF00) | ((rawBoard >> 16) & 0xFF))
        elif self.index == 9:#EG
            return (((rawBoard >> 16) & 0xFF00) | ((rawBoard >> 8) & 0xFF))
        elif self.index == 10:#FH
            return (((rawBoard >> 8) & 0xFF00) | ((rawBoard >> 0) & 0xFF))
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

