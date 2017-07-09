from TupleFeature import TupleFeature

class TupleNetwork(object):
    def __init__(self, code_size, feature_level):
        self.code_size = code_size
        self.feature_level = feature_level
        self.featureSet = []
        for i in range(feature_level):
            self.featureSet.append(TupleFeature(self.code_size, i))
    
    def GetValue(self, state):
        sum = 0;
        for i in range(len(self.featureSet)):
            sum += self.featureSet[i].GetScore(state)
        return sum / len(self.featureSet)

    def UpdateValue(self, state, delta):
        for i in range(len(self.featureSet)):
            self.featureSet[i].UpdateScore(state, delta)
            

