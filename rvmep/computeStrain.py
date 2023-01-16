import numpy as np

def computeDeformationTensor(ref, moving):
    E = np.zeros((ref.n_cells, 3, 3))
    triangles = ref.faces.reshape((-1, 4))[:, 1:]
    D = np.array([[-1, 1, 0], [-1, 0, 1]])
    for i, t in enumerate(triangles):
        E[i] = np.linalg.pinv(D@ ref.points[t]) @ D @moving.points[t]
    return E

def computeStrainTensorInfinitessimal(ref, moving):
    strain = np.zeros((ref.n_cells, 3, 3))
    E = computeDeformationTensor(ref, moving)
    for i, t in enumerate(strain):
        strain[i] = (E[i].T + E[i])/2  - np.eye(3)
    return strain

def computeStrainTensorGreen(ref, moving):
    E = computeDeformationTensor(ref, moving)
    strain = (np.einsum('nij,nkj->nik', E, E) - np.eye(3))/2
    return strain


def computeStrainCoefficientAlongDirection(strain, v):
    """
    Compute the green strain along a direction, with the correction to make it comparable to engineering strain.
    """
    return np.sqrt(1 + 2*np.einsum('ni,nij,nj->n', v, strain, v)) -1 