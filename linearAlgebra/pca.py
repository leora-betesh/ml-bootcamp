import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

"""
I wrote this function just for myself to understand the steps of PCA.   
It isn't written with efficiency in mind.
Tested on iris dataset
"""
def pca(m):
    # 1. Normalize - center the matrix at 0
    m_mean = np.mean(m,axis=0)
    m_centered = m - m_mean.T
    
    # 2. covariance matrix is X.T@X / (number samples -1)
    m_cov = np.dot(m_centered.T,m_centered) / (m.shape[0]-1)

    # 3. Use numpy function to get eigen values and vectors.
    evals, evecs = np.linalg.eigh(m_cov)
    
    # 4. Sort in descending according to eigenvalues
    sorted_indexes = np.argsort(evals)[::-1]
    evecs = evecs[:,sorted_indexes]
    evals = evals[sorted_indexes]
    
    # 5. Not sure why this transpose is here.  Added it to match scikit-learn output
    evecs = evecs.T
    
    return evals, evecs

iris_data = load_iris().data
eigenval_mine, eigenvec_mine = pca(iris_data)
sklearn_pca = PCA().fit(iris_data)

print("my eigen values\n",eigenval_mine)
print("sklearn eigen values\n",sklearn_pca.explained_variance_)
print("my eigen vectors\n",eigenvec_mine)
print("sklearn eigen vectors\n",sklearn_pca.components_)
