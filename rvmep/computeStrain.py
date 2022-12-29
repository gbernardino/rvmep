import numpy as np

def computeDeformationTensor(ref, moving):
    E = np.zeros((ref.n_cells, 3, 3))
    triangles = ref.faces.reshape((-1, 4))[:, 1:]
    for i, t in enumerate(triangles):
        E[i] = np.linalg.pinv(ref.points[t])@moving.points[t]
    return E


def computeStrainTensorGreen(ref, moving):
    strain = np.zeros((ref.n_cells, 3, 3))
    triangles = ref.faces.reshape((-1, 4))[:, 1:]
    for i, t in enumerate(triangles):
        E = np.linalg.pinv(ref.points[t])@moving.points[t]
        strain[i] = (E.T@E - np.eye(3))/2
    return strain

def computeStrainTensorInfinitessimal(ref, moving):
    strain = np.zeros((ref.n_cells, 3, 3))
    triangles = ref.faces.reshape((-1, 4))[:, 1:]
    for i, t in enumerate(triangles):
        E = np.linalg.pinv(ref.points[t])@moving.points[t]
        strain[i] = (E.T + E)/2  - np.eye(3)
    return strain


def computeStrainCoefficientAlongDirection(strain, v):
    """
    Compute the green strain along a direction, with the correction to make it comparable to engineering strain.
    """
    return np.sqrt(1 + 2*np.einsum('ni,nij,nj->n', v, strain, v)) -1 