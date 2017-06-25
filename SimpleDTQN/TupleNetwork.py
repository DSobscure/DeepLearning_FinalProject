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

class TwoStageTupleNetwork(object):
    def __init__(self, bit, feature_count, score_boundary):
        self.feature_count = feature_count
        self.state1_featureSet = []
        self.state2_featureSet = []
        self.score_boundary = score_boundary
        for i in range(self.feature_count):
            self.state1_featureSet.append(N_BitTupleFeature(bit, i))
        for i in range(self.feature_count):
            self.state2_featureSet.append(N_BitTupleFeature(bit, i))
    
    def GetValue(self, rawBoard, score):
        sum = 0;
        for i in range(self.feature_count):
            if score < self.score_boundary:
                sum += self.state1_featureSet[i].GetScore(rawBoard)
            else:
                sum += self.state2_featureSet[i].GetScore(rawBoard)
        return sum

    def UpdateValue(self, rawBoard, delta, score):
        for i in range(self.feature_count):
            if score < self.score_boundary:
                self.state1_featureSet[i].UpdateScore(rawBoard, delta)
            else:
                self.state2_featureSet[i].UpdateScore(rawBoard, delta)
            