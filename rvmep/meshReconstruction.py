"""
Implementation of Linear Surface Reconstruction from Discrete Fundamental Forms on Triangle Meshes,
https://www.cse.msu.edu/~ytong/DiscreteFundamentalForms.pdf
An extension which forces rotation matrices f to be rotation matrices using the matrix logarithm represntation.
"""

import scipy
import numpy as np
try:
    import rotationOperations 
except:
    from . import rotationOperations
import collections

#Utils
import math
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    
    Copied from https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def angle3d(u, v, n):
    """
    return the oriented angle between u and v, using the normal n
    """
    val = np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)
    angle = np.arccos(val)
    sign = np.sign(np.dot(n, np.cross(u, v)))
    return sign * angle

#Main functions for constructing the problem
def computeDihedralAngles(X, normals, edges):
    """
    TODO: vectorize
    """
    dihedralAngles = np.zeros(len(edges))
    for i, e in enumerate(edges):
        dihedralAngles[i] =angle3d(normals[e.f1.fId], normals[e.f2.fId], X[e.edge[1]] - X[e.edge[0]])
    return dihedralAngles

def computeTriangleInteriorAngles(a):
    """
    Given the triangle coordinates, compute its angles.
    """
    angles = np.zeros([len(a), 3])
    for i in range(3):
        j = (i + 1) % 3
        k = (i - 1) % 3
        v1 = (a[:, j] - a[:, i])
        v2 = (a[:, k] - a[:, i])
        v1 /= np.linalg.norm(v1, axis = 1)
        v2 /= np.linalg.norm(v2, axis = 1)
        angles[i,:] = np.arccos(np.sum(v1*v2, axis = 0))
    return angles

def computeRotations(edges, dihedralAngles, a):
    """
    Computes the rotation for each edge. It depends both on the I (interior angles), and the II (dihedral angle)
    
    TODO: vectorize
    """
    R = np.zeros([len(edges), 3, 3])
    e_z = np.array([0,0,1])
    e_x = np.array([1, 0, 0])
    for i, e in enumerate(edges):
        f1 = e.f1.fId
        f2 =e.f2.fId
        theta_T1 =np.arctan2(a[f1, e.f1.pEnd][1] - a[f1, e.f1.pBegin][1],
                                a[f1, e.f1.pEnd][0] - a[f1, e.f1.pBegin][0])
        theta_T2 =np.arctan2(a[f2, e.f2.pEnd][1] - a[f2, e.f2.pBegin][1], 
                                a[f2, e.f2.pEnd][0] - a[f2, e.f2.pBegin][0])
        
        #a2 = a[f1, e.f1.pEnd] - a[f1, e.f1.pBegin]

        R[i] = rotation_matrix(e_z, theta_T1).dot(
            rotation_matrix(e_x, dihedralAngles[i])
        ).dot(rotation_matrix(e_z, -theta_T2))
        
        #return  rotation_matrix(e_z, theta_T1), rotation_matrix(e_x, dihedralAngles[i]), (rotation_matrix(e_z, -theta_T2))
    return R


def reconstructMeshLinear(triangleCoordinates, dihedral, edges, triangles, nPoints):
    R = computeRotations(edges , dihedral, triangleCoordinates)
    
    A_ff, A_fx, A_xx = constructEnergyMatrix( R, triangleCoordinates, edges, triangles, nPoints)
    A_xf = zerosSparse((A_ff.shape[0], A_xx.shape[1]))
    A = scipy.sparse.csc_matrix(scipy.sparse.bmat([[A_xx, A_fx], [A_xf, A_ff]]))
    ATA = A.T.dot(A)
    b = np.zeros((A.shape[1], 3))
    
    b_cnstr = np.concatenate([ np.zeros((1, 3)), np.eye(3)])
    A_cnstr = scipy.sparse.coo_matrix((np.ones(4),
                                   ([0, 1, 2, 3], 
                                    [0, nPoints + 0, nPoints + 1, nPoints + 2])
                                  ), shape = (4, ATA.shape[1])) 
    
    A_lagrange = scipy.sparse.bmat([
                    [ATA, A_cnstr.T], 
                    [A_cnstr, zerosSparse((A_cnstr.shape[0], A_cnstr.shape[0]))]
                ])
    b_lagrange = np.concatenate([b, b_cnstr])
    
    s = scipy.sparse.linalg.spsolve(A_lagrange, b_lagrange)[:- len(b_cnstr), :]
    points = s[:nPoints]
    fs = np.einsum('nij->nji', s[nPoints:, :].reshape((-1, 3, 3)))
    return points, fs, R

#Constructing sparse matrices
def zerosSparse(shape):
    return scipy.sparse.coo_matrix((np.zeros(0),(np.zeros(0), np.zeros(0))), shape =shape)

class SparseMatrixConstructor:
    def __init__(self, nElements):
        self.count = 0
        self.maxElements = nElements
        
        self.data = np.zeros(nElements)
        self.columns = np.zeros(nElements, dtype = np.uint32)
        self.rows = np.zeros(nElements, dtype = np.uint32)
    
    def addElement(self, d, i, j):
        self.data[self.count] = d
        self.columns[self.count] = j
        self.rows[self.count] = i
        self.count += 1
    
    def build(self, shape = None):
        return scipy.sparse.coo_matrix(
            (self.data[:self.count], (self.rows[:self.count], self.columns[:self.count])), 
            shape = shape)

def constructEnergyMatrix( R, a, edges, triangles, nPoints, wRotations = 1., wPoints = 1.):
    """
    Constructs the energy matrix for the quadratic problem. 
    
    TODO: sparse arithmetics from the beginning.
    returns A_xx, A_xf, A_ff st:
    
    A_fk = A_ffk( f )
    A_x = A_xx( x) + A_xf(f)
    """
    #matrix with 3 equation per regular edge (with 2 facesv)    
    nElemsFF =  (4 * 3 * len(edges))
    A_ff_constructor = SparseMatrixConstructor(nElemsFF)
        
    #equations of the form \| f_i - f_j * R_ij \|, for every edge between faces f1 and f2
    #f should  be stored by rows
    nEq = 0
    for indexEdge, e in enumerate(edges):
        #equations relative to rotations
        i,j = e.f1.fId, e.f2.fId
        for k in range(3):
            A_ff_constructor.addElement(-1., nEq, 3* j + k)

            A_ff_constructor.addElement(R[indexEdge, 0, k], nEq, 3 * i + 0)
            A_ff_constructor.addElement(R[indexEdge, 1, k], nEq, 3 * i + 1)
            A_ff_constructor.addElement(R[indexEdge, 2, k], nEq, 3 * i + 2)

            nEq += 1
    A_ff = wRotations * A_ff_constructor.build()
    
    #equations relative to point coordinates
    nEq = 0
    A_fx =  SparseMatrixConstructor(2 * 2 * len(triangles))
    A_xx = SparseMatrixConstructor(2 * 2  *len(triangles)) #Each point appears
    for t, T in enumerate(triangles):
        #For each edge e, with
        for i in range(2):
            p1, p2 = T[i], T[i + 1]
            A_xx.addElement(1. , nEq, p2)
            A_xx.addElement(-1. , nEq, p1)
            
            aV = -a[t][i + 1] + a[t][i]
            A_fx.addElement(aV[0], nEq, 3*t)
            A_fx.addElement(aV[1], nEq, 3*t + 1)

            nEq += 1
    A_fx = wPoints * A_fx.build(shape = [nEq, A_ff.shape[1]])
    A_xx = wPoints * A_xx.build()
    return A_ff, A_fx, A_xx


##
##Using Lie Algebras
##
def getTrianglesToPointsMatrix(m):
    """
    Creates a matrix that goes from the points, stored by rows for each triangle, to the point coordinates.
    """
    P = SparseMatrixConstructor(m.GetNumberOfCells() *3)
    triangles = m.faces.reshape((-1, 4))[:, 1:]
    for i,t in enumerate(triangles):
        P.addElement(1., t[0], 3*i + 0)
        P.addElement(1., t[1], 3*i + 1)
        P.addElement(1., t[2], 3*i + 2)
    return P.build()

def energyRotations(fs, R, edges):
    """
    Computes the energy part of the rotation
    """
    
    return np.linalg.norm(fs[edges[:, 1]] - rotationOperations.matrixProductVectorized(fs[edges[:, 0]], R))**2

def energyPoints(fs, x, As, triangles):
    """
    
    """
    A = np.zeros([3,3])
    A[0,0] = 1
    A[0, 1] = -1
    A[1, 1] = 1
    A[ 1, 2] = -1
    A[2, 0] = 1
    A[2, 2] = -1
    return np.linalg.norm(
        np.einsum('ij, njk-> nik', A, x[triangles]) -
        #np.einsum(' nri,ij,njk ->nrk',fs, A, As)
        np.einsum(' ij,njk, nrk ->nir', A, As,fs)
    )**2


def constructBx(X, As, triangles):
    """
    Constructs such that
    d E_x(f, X)/ d f_log  = -2<d f_i/ d_flog, B >
    
    """
    W = np.zeros([3,3])
    W[0,0] = 1
    W[0, 1] = -1
    W[1, 1] = 1
    W[ 1, 2] = -1
    W[2, 0] = 1
    W[2, 2] = -1
    #return np.einsum('ij,njk,nlk,rl->nir', W, X[triangles], As, W, optimize = True)
    return np.einsum('nji, jk, kl, nlr ->nri', As, W.T, W, X[triangles], optimize = True)


def gradientXVectorized(fs, As, points, triangles, P):
    """
    Very inefficient,.
    
    In order to make it more efficient, compute the matrix that transfer from triangle coordinates to point coordinates.
    """
    W = np.zeros([3,3])
    W[0,0] = 1
    W[0, 1] = -1
    W[1, 1] = 1
    W[ 1, 2] = -1
    W[2, 0] = 1
    W[2, 2] = -1
    
    WTW = W.T.dot(W)
    #d X^2
    temp =  (
                np.einsum('ij,njk->nik', WTW, points[triangles]) - 
             np.einsum('ij, njk, nkr-> nir', WTW, As, np.einsum('nij->nji', fs), optimize = True)
            )
    return 2* P.dot(temp.reshape((-1, 3)))

m0 = np.zeros((3, 3))
m0[1, 2] = -1
m0[2, 1] = 1
m1 = np.zeros((3, 3))
m1[0, 2] = 1
m1[2, 0] = -1
m2 = np.zeros((3,3))
m2[0, 1] = -1
m2[1, 0] = 1
#crossproductMatrix[k].dot(v) = e_k x v
crossProdMatrixAxis = [m0, m1, m2]

def gradientF(fs_log, fs, edges, R, neighboursNP, optimize = True, BsX = 0, onlyX = False):
    """
    It's a fucking mess, but working on vectorizing all operations. Got a 10x acceleration!
    
    BsX is the extra term coming from the points energy, that needs to be added to the left of the dot product <d exp(v)/dv , B>
    Note that with the scalar product that induces the kroennecker norm, for A, B matrices, <A, B> = \sum_k <A e_k, B e_k>. This is actually easier to compute
    [e_k]x v = e_k \cross v
    """

    A = np.zeros([3,9])
    A[0, 7] = 1
    A[0, 5] = -1   
    A[1, 2] = 1
    A[1, 6] = -1
    A[2, 3] = 1
    A[2, 1] = -1  
    fs_log_crossProd_matrix = np.einsum('ij, ki->kj', A,fs_log).reshape((-1, 3, 3))    

    directB = rotationOperations.matrixProductVectorized(fs[edges[:, 0]], R)
    indirectB = rotationOperations.matrixProductVectorized(fs[edges[:, 1]], np.einsum('nij->nji', R))
    
    Bs = np.concatenate([directB, indirectB])
    if onlyX:
        acumB = BsX
    else:
        acumB = np.sum(Bs[neighboursNP], axis = 1) + BsX
    fs_log_outer = np.einsum('ni,nj->nij', fs_log, fs_log)
    norm = (1e-12 + np.einsum('ni,ni->n', fs_log, fs_log)).reshape((-1, 1, 1))
    #Using formula (8) of "A Compact Formula for the Derivative of a 3-D Rotation in Exponential Coordinates
    derivative_2ndFactor = (fs_log_outer + rotationOperations.matrixProductVectorized(np.einsum('nij->nji', fs)  - np.eye(3), fs_log_crossProd_matrix)) / norm
    derivative_2ndFactor[np.where(norm < 1e-6)[0]] = np.eye(3)
    grad = np.zeros_like(fs_log)
    if not optimize:
        for i in range(len(grad)):
           # derivative_2ndFactor = (fs_log_outer[i] + (fs[i].T - np.eye(3)).dot( fs_log_crossProd_matrix[i])) \
            #                        / np.dot(fs_log[i], fs_log[i])
            for k in range(3):
                #grad[i] += -2 * (-fs[i].dot(crossProdMatrix(e_k)).dot(derivative_2ndFactor)).T.dot(acumB_i[:, k])
                grad[i] += -2 * (-fs[i].dot(crossProdMatrixAxis[k]).dot(derivative_2ndFactor[i])).dot(acumB[i,:, k])
    else:
        for k in range(3):
    
            grad += -2 * np.einsum('nji,kj,nwk, nw-> ni',  derivative_2ndFactor,  crossProdMatrixAxis[k], -fs, acumB[:,:, k], optimize = True)
    return grad