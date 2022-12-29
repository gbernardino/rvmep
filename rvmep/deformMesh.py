
import numpy as np
import time
import scipy, scipy.optimize
import logging
from . import meshReconstruction, rotationOperations, edgeStructures

def procrustes(p1, p2):
    R, _ = scipy.linalg.polar((p1 - np.mean(p1, axis = 0)).T.dot(p2 - np.mean(p2, axis = 0)))
    return np.einsum('ij,nj ->ni', R, p2 -   np.mean(p2, axis = 0)) +   np.mean(p1, axis = 0)


#Functions for computing the energy and gradient of the 
def flatten(X, f):
    return np.concatenate([X, f]).reshape(-1)
def unflatten(X, nPoints):
    X_hat = X.reshape((-1, 3))
    p = X_hat[:nPoints]
    f = X_hat[nPoints:]
    return p, f
def energy(X_hat, nPoints, R, edgesNP, As, triangles):
    X = X_hat.reshape((-1, 3))
    p = X[:nPoints, : ]
    f_log = X[nPoints:, :]
    f = rotationOperations.exponentialRotationVectorized(f_log)
    return (
        meshReconstruction.energyRotations(f, R, edgesNP) +
        meshReconstruction.energyPoints(f, p, As, triangles)
    )
def gradient(X_hat, nPoints, R, edgesNP, As, triangles, neighboursNP, P):
    X = X_hat.reshape((-1, 3))
    p = X[:nPoints, : ]
    f_log = X[nPoints:, :]
    f = rotationOperations.exponentialRotationVectorized(f_log)
    BsX = meshReconstruction.constructBx(p, As, triangles)
    gX =meshReconstruction.gradientXVectorized(f, As, p, triangles, P)
    gF = meshReconstruction.gradientF(f_log, f, edgesNP, R, neighboursNP, BsX = BsX)
    return np.concatenate([gX, gF]).reshape(-1)

def deformMeshOptimisation(meshVTK, targetTriangleCoordinates, dihedralAngles, **kwargs):
    """
    Performs the optimisation procedure to deform the meshNP(my kind of pyvista mesh, with some useful functions for treating with fields), to generate a mesh that has the given in-triangle coordinates and dihedral angles
    """

    #Initial guess - WARNING, not always needed
    triangles = meshVTK.faces.reshape((-1, 4))[:, 1:]
    edgesNonDegenerate = edgeStructures.getEdges(meshVTK)

    ps0, fs0, R = meshReconstruction.reconstructMeshLinear(targetTriangleCoordinates, dihedralAngles, edgesNonDegenerate, triangles, meshVTK.GetNumberOfPoints())
    if kwargs.get('reconstructionLinear', False):
        p = ps0
    else:
        if kwargs.get('initial_guess_linear', False):
            f0_rotation = rotationOperations.projectToRotation(fs0)
            f0_log = rotationOperations.logarithmRotationVectorized(f0_rotation)
        else:
            ps0 = meshVTK.points.copy()
            f0_log = np.zeros((meshVTK.GetNumberOfCells(), 3))
        nPoints = meshVTK.GetNumberOfPoints()
        edgesNP, neighboursNP = edgeStructures.getEdgesAndNeighbours(meshVTK, edgesNonDegenerate)
        P = meshReconstruction.getTrianglesToPointsMatrix(meshVTK)
        t = time.time()
        nPoints = meshVTK.GetNumberOfPoints()
        xBFGS = scipy.optimize.minimize(lambda X: energy(X, nPoints, R, edgesNP, targetTriangleCoordinates, triangles),
                                x0 = np.concatenate([ps0, f0_log]).reshape(-1),
                                jac = lambda X: gradient(X, nPoints, R, edgesNP, targetTriangleCoordinates, triangles, neighboursNP, P), method = 'L-BFGS-B'
                               )
        logging.info( 'Time needed reconstruction = %f' % (time.time() - t))
        p, f = unflatten(xBFGS.x, meshVTK.GetNumberOfPoints())
    pointsAligned = procrustes(meshVTK.points, p)
    return pointsAligned
    
def triangleCoordinatesFromLengths(ls):
    """
    Given the lenghts, generate the triangle coordinates. The lenghts are defined by
    l_0 = | p_1 - p_0|
    l_1 = | p_2 - p_1|
    l_2 = | p_2 - p_0|
    
    The triangle coordinates are defined as:
    a_0 = (0,0,0 )
    a_1 = (l_0, 0, 0)  [It lies in the x-axis]
    a_2 = (x, y, 0)    [Needs to be computed using the equations]
    """
    A = np.zeros((3,3))
    A[1,0] = ls[0]
    # Get the coordinates of a_2 solving two equations
    # x^2 + y^2 = l_2^2
    # (l_0 -x)^ 2 + y^2 = l_1^2
    # x_2^2 - (l_0 -x)^ 2 = l2^2 - l_1^2
    # 2*l_0*x  =  l2^2 - l_1^2 + l_0^2
    x = (ls[0]**2 +ls[2]**2 - ls[1]**2)/(2 * ls[0])
    y = np.sqrt(ls[2]**2 - x**2)
    A[2,0] = x
    A[2, 1] = y
    return A
