import numpy as np

def computeStrainTensorGreen(ref, moving):
    strain = np.zeros((ref.n_cells, 3, 3))
    triangles = ref.faces.reshape((-1, 3))[:, 1:]
    for i, t in enumerate(triangles):
        E = np.pinv(ref.points[t])@moving.points[t]
        strain[i] = (E.T@E - np.eye(3))/2
    return strain

def computeStrainTensorInfinitessimal(ref, moving):
    strain = np.zeros((ref.n_cells, 3, 3))
    triangles = ref.faces.reshape((-1, 3))[:, 1:]
    for i, t in enumerate(triangles):
        E = np.pinv(ref.points[t])@moving.points[t]
        strain[i] = (E.T + E)/2  - np.eye(3)
    return strain


def computeStrainCoefficientAlongDirection(strain, v):
    """
    Compute the green strain along a direction, with the correction to make it comparable to engineering strain.
    """
    return = np.sqrt(1 + 2*np.einsum('ni,nij,nj->n', v, strain, v) -1 