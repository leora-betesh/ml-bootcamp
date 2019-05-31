import numpy as np
from sklearn.decomposition import PCA

"""
I wrote this function just for myself to understand the steps of PCA.   It isn't written with efficiency in mind.
"""
def pca(m):
    # 1. Subtract the column means to center the matrix at 0
    m_mean = np.mean(m,axis=0)
    m_centered = m - m_mean.T

    # 2. covariance matrix is X.T@X / (number samples -1)
    m_cov = np.dot(m_centered.T,m_centered) / (m.shape[0]-1)

    # 3. use numpy eigh function to get eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(m_cov)

    # 4. Sort the eigenvectors and eigenvalues in descending order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]

    return evals,evecs.T

M1 = np.array([[1,2,3,4],[5,8,11,14],[2,4,6,8],[5,10,15,20]])
eigenval_mine, eigenvec_mine = pca(M1)
sklearn_pca = PCA().fit(M1)
print(eigenval_mine)
print(sklearn_pca.explained_variance_)
print("_"*30)
print(eigenvec_mine)
print(sklearn_pca.components_)
