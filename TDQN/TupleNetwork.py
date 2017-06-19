from TupleFeature16bit import TupleFeature16bit

class TupleNetwork(object):
    def __init__(self):
        self.featureSet = []

        for i in range(10):
            self.featureSet.append(TupleFeature16bit(i + 1))
    
    def GetValue(self, rawBoard):
        sum = 0;
        for i in range(len(self.featureSet)):
            sum += self.featureSet[i].GetScore(rawBoard)
        return sum

    def UpdateValue(self, rawBoard, delta):
        for i in range(len(self.featureSet)):
            self.featureSet[i].UpdateScore(rawBoard, delta)

