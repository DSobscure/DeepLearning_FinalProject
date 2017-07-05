from TupleFeature import TupleFeature

class TupleNetwork(object):
    def __init__(self):
        self.featureSet = []
        self.featureSet.append(TupleFeature(1))
        self.featureSet.append(TupleFeature(2))
        self.featureSet.append(TupleFeature(3))
        self.featureSet.append(TupleFeature(4))
        self.featureSet.append(TupleFeature(5))
        self.featureSet.append(TupleFeature(6))
    
    def GetValue(self, raw_state):
        sum = 0;
        for i in range(1):
            state = (raw_state >> (i * 24)) & 0xFFFFFFFFFFFF
            for j in range(len(self.featureSet)):
                sum += self.featureSet[j].GetScore(state)
        return sum

    def UpdateValue(self, raw_state, delta):
        for i in range(1):
            state = (raw_state >> (i * 24)) & 0xFFFFFFFFFFFF
            for j in range(len(self.featureSet)):
                self.featureSet[j].UpdateScore(state, delta)

