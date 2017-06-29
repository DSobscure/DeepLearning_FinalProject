from TupleFeature8bit_FlappyBird import TupleFeature8bit_FlappyBird
from TupleFeature16bit_FlappyBird import TupleFeature16bit_FlappyBird

class TupleNetwork_FlappyBird(object):
    def __init__(self):
        self.featureSet = []
        self.featureSet.append(TupleFeature8bit_FlappyBird(1))
        self.featureSet.append(TupleFeature8bit_FlappyBird(2))
        self.featureSet.append(TupleFeature8bit_FlappyBird(3))
        self.featureSet.append(TupleFeature16bit_FlappyBird(1))
        self.featureSet.append(TupleFeature16bit_FlappyBird(2))
    
    def GetValue(self, rawBoard):
        sum = 0;
        for i in range(len(self.featureSet)):
            sum += self.featureSet[i].GetScore(rawBoard)
        return sum

    def UpdateValue(self, rawBoard, delta):
        for i in range(len(self.featureSet)):
            self.featureSet[i].UpdateScore(rawBoard, delta)

