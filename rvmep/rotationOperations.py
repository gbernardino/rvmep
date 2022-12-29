import numpy as np
import scipy

def projectToRotation(fs):
    """
    Potentially slow
    """
    fsR = np.zeros(fs.shape)
    for i, f in enumerate(fs):
        fsR[i], _ = scipy.linalg.polar(f)
    return fsR

def logarithmRotationVectorized(R):
    A = np.zeros([3,9])
    A[0, 7] = 1
    A[0, 5] = -1   
    A[1, 2] = 1
    A[1, 6] = -1
    A[2, 3] = 1
    A[2, 1] = -1  
    theta = np.arccos((R[:,0,0] + R[:, 1,1] + R[:, 2, 2] - 1)/2)[:,np.newaxis]
    return theta/(2 * (np.sin(theta) + 1e-6)) * np.einsum('ij, kj->ki', A, R.reshape((-1, 9)))

def findOptimalRotation(X,As, triangles):
    W = np.zeros([3,3])
    W[0,0] = 1
    W[0, 1] = -1
    W[1, 1] = 1
    W[ 1, 2] = -1
    W[2, 0] = 1
    W[2, 2] = -1
    fs = np.zeros([len(triangles), 3, 3])
    for i, t in enumerate(triangles):
        fs[i], _ = scipy.linalg.polar(X[t].T.dot(W.T).dot(W).dot(As[i]))
    return fs

def exponentialRotationVectorized(V):
    """
    Maybe need to accelerate a bit
    """
    A = np.zeros([9, 3])
    A[1, 2] = -1
    A[2, 1] = 1
    A[3, 2] = 1
    A[5, 0] = -1
    A[6, 1] = -1
    A[7, 0] = 1
    I = np.eye(3).reshape(-1)
    theta = np.linalg.norm(V, axis = 1)[:, np.newaxis]
    v = V/(theta + 1e-9)
    return (I + np.sin(theta) *  np.einsum('ij, kj->ki', A, v) \
            + (1 - np.cos(theta)) *(np.einsum('ki, kj ->kij', v, v).reshape((-1, 9)) - I )).reshape((-1, 3,3))

def matrixProductVectorized(A, B):
    return np.einsum('nij, njk -> nik', A, B,  optimize = True)

def crossProdMatrix(a):
    """
    TODO: make vectorizable, with An and reshape
    """
    A = np.zeros((3, 3))
    A[0, 1] = - a[2]
    A[0, 2] = a[1]
    A[1,0] = a[2]
    A[1, 2] = - a[0]
    A[2, 0] = -a[1]
    A[2, 1] = a[0]
    return A

def crossProdMatrixVectorized(a):
    """
    """
    A = np.zeros([9, 3])
    A[1, 2] = -1
    A[2, 1] = 1
    A[3, 2] = 1
    A[5, 0] = -1
    A[6, 1] = -1
    A[7, 0] = 1
    
    return np.einsum('ij,nj->ni' , A, a).reshape((-1, 3,3))