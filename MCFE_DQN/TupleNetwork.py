from TupleFeature import TupleFeature

class TupleNetwork(object):
    def __init__(self):
        self.featureSet = []
        self.featureSet.append(TupleFeature(1))
        self.featureSet.append(TupleFeature(2))
        self.featureSet.append(TupleFeature(3))
        self.featureSet.append(TupleFeature(4))
        self.featureSet.append(TupleFeature(5))
    
    def GetValue(self, rawBoard):
        sum = 0;
        for i in range(len(self.featureSet)):
            sum += self.featureSet[i].GetScore(rawBoard)
        return sum

    def UpdateValue(self, rawBoard, delta):
        for i in range(len(self.featureSet)):
            self.featureSet[i].UpdateScore(rawBoard, delta)

