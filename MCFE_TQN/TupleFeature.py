import numpy as np

class TupleFeature(object):
    def __init__(self, bit, index):
        self.bit = bit
        self.index = index
        self.tuples = np.zeros(2 ** self.bit)
        self.mask = 2 ** self.bit - 1
    def GetIndex(self, rawBoard):
        return (rawBoard >> (self.index*self.bit)) & self.mask
    
    def UpdateScore(self, rawBoard, delta):
        self.tuples[self.GetIndex(rawBoard)] += delta
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

