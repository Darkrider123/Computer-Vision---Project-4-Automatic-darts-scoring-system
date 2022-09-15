from sklearn.cluster import MeanShift

def clusterize(data):
    model = MeanShift(bandwidth=30 ,cluster_all=False).fit(data)
    return model.labels_