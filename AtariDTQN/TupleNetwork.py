from TupleFeature12bit import TupleFeature12bit
from TupleFeature16bit import TupleFeature16bit

class TupleNetwork(object):
    def __init__(self):
        self.featureSet = []
        self.featureSet.append(TupleFeature12bit(1))
        self.featureSet.append(TupleFeature12bit(2))

        self.featureSet.append(TupleFeature16bit(1))
        self.featureSet.append(TupleFeature16bit(2))
    
    def GetValue(self, rawBoard):
        sum = 0;
        for i in range(len(self.featureSet)):
            sum += self.featureSet[i].GetScore(rawBoard)
        return sum

    def UpdateValue(self, rawBoard, delta):
        for i in range(len(self.featureSet)):
            self.featureSet[i].UpdateScore(rawBoard, delta)

