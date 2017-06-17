import numpy as np

class FeatureTable(object):
    def __init__(self, code_size, feature_count):
        self.code_size = code_size
        self.feature_count = feature_count
        self.featureSet = []
        for i in range(feature_count):
            self.featureSet.append(np.zeros(2 ** self.code_size))
    
    def GetValue(self, feature_code_set):
        sum = 0;
        for i in range(self.feature_count):
            sum += self.featureSet[i][feature_code_set[i]]
        return sum

    def UpdateValue(self, feature_code_set, delta):
        for i in range(self.feature_count):
            self.featureSet[i][feature_code_set[i]] += delta