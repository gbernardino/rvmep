import scipy, scipy.sparse,scipy.sparse.linalg, numpy as np

def computeAnatomicalDirectionsHeatEquation(mesh, apexPointId, valvesPointsId):
    """
    Computes the longitudinal direction, using a single orifice.
    """
    triangles = mesh.faces.reshape((-1, 4))[:, 1:]
    boundary = {v: 1. for v in valvesPointsId}
    boundary[apexPointId] = 0
    heat = solveLaplaceBeltrami(mesh.points, triangles, boundary)
    vLongitudinal = grad_3d(mesh, heat)
    vLongitudinal = vLongitudinal / np.linalg.norm(vLongitudinal, axis = 1).reshape((-1, 1))
    vCircumferential = np.cross(vLongitudinal, mesh.cell_normals)
    return vLongitudinal, vCircumferential


def solveLaplaceBeltrami(points, triangles, boundary):
    """
    Solve laplace-beltrami equation with boundary conditions 
    """
    nPoints = len(points)
    nConstraints = len(boundary)
    L = ComputeCotangentLaplacian( points.T, triangles.T)
    lagrange_multipliers = np.zeros((len(boundary), nPoints))
    boundary_values = np.zeros(nConstraints)
    for i, (k, v) in enumerate(boundary.items()):  
        lagrange_multipliers[i, k] = 1
        boundary_values[i] = v
    zeros = np.zeros((nConstraints, nConstraints))
    L_hat = scipy.sparse.bmat([[L, lagrange_multipliers.T], [lagrange_multipliers, zeros]])
    b_hat = np.concatenate([np.zeros(nPoints), boundary_values])
    x_hat = scipy.sparse.linalg.spsolve(L_hat, b_hat)
    return x_hat[:nPoints]

def ComputeCotangentLaplacian( vertex, faces ):
    """
    Calculates the laplacian of a mesh
    vertex 3xN numpy.array: vertices
    faces 3xM numpy.array: faces
    Return the Laplacian (sparse matrix probably).
    """
    n = vertex.shape[1]
    m = faces.shape[1]
    
    #compute mesh weight matrix
    W = scipy.sparse.coo_matrix((n,n))
    for i in np.arange(1,4,1):
        i1 = np.mod(i-1,3)
        i2 = np.mod(i  ,3)
        i3 = np.mod(i+1,3)
        pp = vertex[:,faces[i2,:]] - vertex[:,faces[i1,:]]
        qq = vertex[:,faces[i3,:]] - vertex[:,faces[i1,:]]
        #% normalize the vectors
        pp = pp / np.sqrt(np.sum(pp**2, axis=0))
        qq = qq / np.sqrt(np.sum(qq**2, axis=0))

        #% compute angles
        ang = np.arccos(np.sum(pp*qq, axis=0))
        W = W + scipy.sparse.coo_matrix( (1 / np.tan(ang),(faces[i2,:],faces[i3,:])), shape=(n, n) )
        W = W + scipy.sparse.coo_matrix( (1 / np.tan(ang),(faces[i3,:],faces[i2,:])), shape=(n, n) )


    #compute laplacian
    d = W.sum(axis=0)
    D = scipy.sparse.dia_matrix((d, 0), shape=(n,n) )
    L = D - W

    return L

def propagateDirections(meshRef, directions, meshTarget):
    pass


# Utils
# Implemented by Thomas D'Argent
def grad_3d(mesh, scalarField):
    cell_gradients = np.zeros((mesh.n_cells, 3))
    triangles = mesh.faces.reshape((-1, 4))[:, 1:]

    for i, (p0_i, p1_i, p2_i) in enumerate(triangles):
        p10 = mesh.points[p1_i] - mesh.points[p0_i]
        p20 = mesh.points[p2_i] - mesh.points[p0_i]
        
        gradient = np.zeros(3)
        # compute the difference in value between p{1,2} and p0
        t10 = scalarField[p1_i] - scalarField[p0_i]
        t20 = scalarField[p2_i] - scalarField[p0_i]

        # t10 = gradient@p10 (resp. t20 and p20)
        gradient = np.linalg.pinv([p10, p20, np.cross(p10, p20)])@np.array([t10, t20, 0])
        assert np.abs(gradient@p10 - t10) <= 1e-4
        assert np.abs(gradient@p20 - t20) <= 1e-4
        assert np.abs(gradient@np.cross(p10, p20)) <= 1e-4

        cell_gradients[i] = gradient
    return cell_gradients