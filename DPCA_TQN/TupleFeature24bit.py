import numpy as np

class TupleFeature24bit(object):
    def __init__(self, index):
        self.index = index
        self.tuples = np.zeros(2 ** 24)
    #12bits as a block total 48bit
	#AB
    #CD
    #EF
    #GH
    def GetIndex(self, rawBoard):
        if self.index == 1:#AB
            return (rawBoard >> 72) & 0xFFFFFF
        elif self.index == 2:#CD
            return (rawBoard >> 48) & 0xFFFFFF
        elif self.index == 3:#EF
            return (rawBoard >> 24) & 0xFFFFFF
        elif self.index == 4:#GH
            return (rawBoard >> 0) & 0xFFFFFF
        elif self.index == 5:#AC
            return ((rawBoard >> 72) & 0xFFF000) | ((rawBoard >> 60) & 0xFFF)
        elif self.index == 6:#BD
            return ((rawBoard >> 60) & 0xFFF000) | ((rawBoard >> 48) & 0xFFF)
        elif self.index == 7:#CE
            return ((rawBoard >> 48) & 0xFFF000) | ((rawBoard >> 36) & 0xFFF)
        elif self.index == 8:#DF
            return ((rawBoard >> 36) & 0xFFF000) | ((rawBoard >> 24) & 0xFFF)
        elif self.index == 9:#EG
            return ((rawBoard >> 24) & 0xFFF000) | ((rawBoard >> 12) & 0xFFF)
        elif self.index == 10:#FH
            return ((rawBoard >> 12) & 0xFFF000) | ((rawBoard >> 0) & 0xFFF)
        else:
            return 0
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

