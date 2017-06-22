from N_BitTupleFeature import N_BitTupleFeature

class TupleNetwork(object):
    def __init__(self, bit, feature_count):
        self.featureSet = []
        for i in range(feature_count):
            self.featureSet.append(N_BitTupleFeature(bit, i))
    
    def GetValue(self, rawBoard):
        sum = 0;
        for i in range(len(self.featureSet)):
            sum += self.featureSet[i].GetScore(rawBoard)
        return sum

    def UpdateValue(self, rawBoard, delta):
        for i in range(len(self.featureSet)):
            self.featureSet[i].UpdateScore(rawBoard, delta)
