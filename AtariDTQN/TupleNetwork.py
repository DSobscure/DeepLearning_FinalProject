from TupleFeature16bit import TupleFeature16bit
from TupleFeature24bit import TupleFeature24bit

class TupleNetwork(object):
    def __init__(self):
        self.featureSet = []
        #self.featureSet.append(TupleFeature16bit(1))
        #self.featureSet.append(TupleFeature16bit(2))

        self.featureSet.append(TupleFeature24bit(1))
        self.featureSet.append(TupleFeature24bit(2))
        self.featureSet.append(TupleFeature24bit(3))
        self.featureSet.append(TupleFeature24bit(4))
        self.featureSet.append(TupleFeature24bit(5))
        self.featureSet.append(TupleFeature24bit(6))
    
    def GetValue(self, rawBoard):
        sum = 0;
        for i in range(len(self.featureSet)):
            sum += self.featureSet[i].GetScore(rawBoard)
        return sum

    def UpdateValue(self, rawBoard, delta):
        for i in range(len(self.featureSet)):
            self.featureSet[i].UpdateScore(rawBoard, delta)

