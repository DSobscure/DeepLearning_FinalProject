import numpy as np

class TupleFeature(object):
    def __init__(self, bit, index):
        self.bit = bit
        self.index = index
        self.tuples = np.zeros(2 ** self.bit)
        self.tcl_sum = np.zeros(2 ** self.bit)
        self.tcl_abs_sum = np.zeros(2 ** self.bit)
        self.mask = 2 ** self.bit - 1
    def GetIndex(self, rawBoard):
        return (rawBoard >> (self.index*self.bit)) & self.mask
    
    def UpdateScore(self, rawBoard, delta):
        index = self.GetIndex(rawBoard)
        if self.tcl_abs_sum[index] == 0:
            self.tuples[index] += delta
        else:
            self.tuples[index] += delta * abs(self.tcl_sum[index]) / self.tcl_abs_sum[index]
        
        self.tcl_sum[index] += delta
        self.tcl_abs_sum[index] += abs(delta)
    
    def GetScore(self, rawBlock):
        return self.tuples[self.GetIndex(rawBlock)]

