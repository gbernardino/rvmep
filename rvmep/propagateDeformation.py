from . import anatomicalDirections, computeStrain, meshReconstruction, tools, deformMesh, edgeStructures
import  numpy as np

def computeStrainAnatomic(ref, moving):
    apexId, pointsTricuspid, pointsPulmonary = tools.getTomtecApexValvePointsRV()
    valvePoints = np.concatenate([pointsTricuspid, pointsPulmonary])
    vLongitudinal, vCirc = anatomicalDirections.computeAnatomicalDirectionsHeatEquation(ref, apexId, valvePoints, True)
    S = computeStrain.computeStrainTensorGreen(ref, moving)
    return computeStrain.computeStrainCoefficientAlongDirection(S, vLongitudinal), computeStrain.computeStrainCoefficientAlongDirection(S, vCirc)

def getCoordinatesTriangle(vtkMesh, triangleCoordinates2D):
    triangleCoordinates3D = np.zeros((vtkMesh.GetNumberOfCells(), 3, 3))
    for t, A in enumerate(triangleCoordinates2D):
        ls = [  
                np.linalg.norm(A[1] - A[0]), 
                np.linalg.norm(A[2] - A[1]),
                np.linalg.norm(A[2] - A[0])
            ]
        triangleCoordinates3D[t] = deformMesh.triangleCoordinatesFromLengths(ls)
    return triangleCoordinates3D

def getDeformationAnatomic(ref_undeformed, ref_deformed, anatomicalLandmarks, useDirectionsRef = True):
    apexId, valvePoints = anatomicalLandmarks['apexId'], anatomicalLandmarks['valvePoints']
    vLongitudinal_ref, vCirc_ref = anatomicalDirections.computeAnatomicalDirectionsHeatEquation(ref_undeformed, apexId, valvePoints, True)
    systemOfCoordinatesRef = np.stack([vLongitudinal_ref, vCirc_ref],axis = 1)
    systemOfCoordinatesDef = anatomicalDirections.propagateDirections(ref_undeformed, systemOfCoordinatesRef , ref_deformed)

    E = computeStrain.computeDeformationTensor(ref_undeformed, ref_deformed)
    # Warning, not sure if it should be multiplied by the system of coordinates at ref, or at deformed
    if useDirectionsRef:
        E_anatomic = np.einsum('nij,njk,nhk-> nih', systemOfCoordinatesRef,  E, systemOfCoordinatesRef)
    else:
        E_anatomic = np.einsum('nij,njk,nhk-> nih', systemOfCoordinatesDef,  E, systemOfCoordinatesRef)

    for i, e in enumerate(E_anatomic):
        if np.any(np.isnan(e)):
            E_anatomic[i] = np.eye(2)
    return E_anatomic

def getLengthsDeformation(mesh, G_anatomic):
    """
    Apply the deformation and get the lengths
    """
    vLongitudinal_ref = mesh.cell_data['vLong']
    vCirc_ref = mesh.cell_data['vCirc']


    systemOfCoordinatesRef = np.stack([vLongitudinal_ref, vCirc_ref],axis = 1)
    triangles = mesh.faces.reshape((-1, 4))[:, 1:]
    edgeLength = np.zeros((mesh.GetNumberOfCells(), 3))
    edgeDirection =  np.zeros((mesh.GetNumberOfCells(), 3, 2))
    # Get edges lengths
    for i in range(3):
        edgeLength[:,i] = np.linalg.norm(mesh.points[triangles[:, i]] - mesh.points[triangles[:, (i + 1)%3]], axis = 1)
        edgeDirection[:,i] = np.einsum('nij, nj->ni', systemOfCoordinatesRef,  mesh.points[triangles[:, i]] - mesh.points[triangles[:, (i + 1)%3]])
        edgeDirection[:, i] /= edgeLength[:, i].reshape((-1, 1))
        elongation = computeStrain.computeStrainCoefficientAlongDirection(G_anatomic, edgeDirection[:, i]) + 1
        edgeLength[:,i] *= elongation
    return edgeLength

def translateDeformation(target, ref_undeformed, ref_deformed, linear =False, anatomicalLandmarks = None, useDirectionsPrecomputed = False):
    """
    Propagate the same deformation from ref undeformed to deformed to the target mesh.
    Uses the Green strain tensor to modify the triangle, and get the new edge lengths
    """
    if not useDirectionsPrecomputed:
        if anatomicalLandmarks is None:
            apexId, pointsTricuspid, pointsPulmonary = tools.getTomtecApexValvePointsRV()
            valvePoints = np.concatenate([pointsTricuspid, pointsPulmonary])
            anatomicalLandmarks = {'apexId' : apexId, 'valvePoints' : valvePoints}
        else:
            apexId, valvePoints = anatomicalLandmarks['apexId'], anatomicalLandmarks['valvePoints']

        vLongitudinal_ref, vCirc_ref = anatomicalDirections.computeAnatomicalDirectionsHeatEquation(ref_undeformed, apexId, valvePoints, True)
        vLongitudinal_tgt, vCirc_tgt = anatomicalDirections.computeAnatomicalDirectionsHeatEquation(target, apexId, valvePoints, True)
        target.cell_data.set_array(vLongitudinal_tgt, 'vLong')
        target.cell_data.set_array(vCirc_tgt, 'vCirc')

    else:
        vLongitudinal_ref = ref_undeformed.cell_data['vLong']
        vCirc_ref = ref_undeformed.cell_data['vCirc']

    systemOfCoordinatesRef = np.stack([vLongitudinal_ref, vCirc_ref],axis = 1)

    # Compute the new triangle coordinates, using the edges lengths.
        # First, compute the deformation tensor, and express it in anatomical coordinates
    G =  computeStrain.computeStrainTensorGreen(ref_undeformed, ref_deformed)
    G_anatomic =  np.einsum('nij,njk,nmk->nim', systemOfCoordinatesRef , G, systemOfCoordinatesRef)
        # Then obtain for each triangle, the lengths of the edges, after being deformed using G.
    newEdgeLength = getLengthsDeformation(target, G_anatomic)      
        # Finally, compute the triangle coordinates from the deformed edges lengths
    triangleCoordinates3D = deformMesh.triangleCoordinatesFromLengthsVector(newEdgeLength)

    edges = edgeStructures.getEdges(target)

    dihedralsTarget = meshReconstruction.computeDihedralAngles(target.points, target.cell_normals, edges)
    dihedrals0 = meshReconstruction.computeDihedralAngles(ref_undeformed.points, ref_undeformed.cell_normals, edges)
    dihedralsT = meshReconstruction.computeDihedralAngles(ref_deformed.points, ref_deformed.cell_normals, edges)
    delta_dihedrals = dihedralsT - dihedrals0
    if not linear:
        return deformMesh.deformMeshOptimisation(target, triangleCoordinates3D, delta_dihedrals  + dihedralsTarget)
    else:
        triangles= target.faces.reshape((-1, 4))[:, 1:]
        return meshReconstruction.reconstructMeshLinear(triangleCoordinates3D, delta_dihedrals + dihedralsTarget, edges, triangles, target.GetNumberOfPoints())

def applyDeformation(target, E_anatomic, delta_dihedrals, anatomicalLandmarks, edges, linear = False):
    apexId, valvePoints = anatomicalLandmarks['apexId'], anatomicalLandmarks['valvePoints']
    triangles= target.faces.reshape((-1, 4))[:, 1:]
    edges = edgeStructures.getEdges(target)

    pointsRef =  target.points[triangles]

    vLongitudinal, vCirc = anatomicalDirections.computeAnatomicalDirectionsHeatEquation(target, apexId, valvePoints, True)
    systemOfCoordinatesTarget= np.stack([vLongitudinal, vCirc],axis = 1)

    dihedralsRef = meshReconstruction.computeDihedralAngles( target.points,  target.cell_normals, edges)
    dihedrals_transformed = dihedralsRef + (delta_dihedrals)

    coordinatesAnatomic = np.einsum('nij, nkj->nik', pointsRef, systemOfCoordinatesTarget)
    coordinatesAnatomic_transformed =np.einsum('nij, nkj->nki', E_anatomic, coordinatesAnatomic)
    triangleCoordinates3D = getCoordinatesTriangle(target, coordinatesAnatomic_transformed)
    if not linear:
        return deformMesh.deformMeshOptimisation(target, triangleCoordinates3D, dihedrals_transformed)
    else:
        return meshReconstruction.reconstructMeshLinear(triangleCoordinates3D, dihedrals_transformed, edges, triangles, target.GetNumberOfPoints())