from FourTupleFeature import FourTupleFeature

class TupleNetwork(object):
    def __init__(self):
        self.featureSet = []
        self.featureSet.append(FourTupleFeature(1))
        self.featureSet.append(FourTupleFeature(2))
        self.featureSet.append(FourTupleFeature(3))
        self.featureSet.append(FourTupleFeature(4))
        self.featureSet.append(FourTupleFeature(5))
    
    def GetValue(self, rawBoard):
        sum = 0;
        for i in range(len(self.featureSet)):
            sum += self.featureSet[i].GetScore(rawBoard)
        return sum

    def UpdateValue(self, rawBoard, delta):
        for i in range(len(self.featureSet)):
            self.featureSet[i].UpdateScore(rawBoard, delta)

