from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
特征降维
"""


def pca(data):
    return PCA(n_components=2).fit_transform(data)


def lds(data, target):
    return LatentDirichletAllocation(n_components=2).fit_transform(data, data[target])
