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

def translateDeformation(target, ref_undeformed, ref_deformed, linear =False):
    apexId, pointsTricuspid, pointsPulmonary = tools.getTomtecApexValvePointsRV()
    valvePoints = np.concatenate([pointsTricuspid, pointsPulmonary])
    edges = edgeStructures.getEdges(target)

    vLongitudinal, vCirc = anatomicalDirections.computeAnatomicalDirectionsHeatEquation(target, apexId, valvePoints, True)
    systemOfCoordinatesTarget= np.stack([vLongitudinal, vCirc],axis = 1)

    vLongitudinal_ref, vCirc_ref = anatomicalDirections.computeAnatomicalDirectionsHeatEquation(ref_undeformed, apexId, valvePoints, True)
    systemOfCoordinatesRef = np.stack([vLongitudinal_ref, vCirc_ref],axis = 1)

    E = computeStrain.computeDeformationTensor(ref_undeformed, ref_deformed)
    E_anatomic = np.einsum('nij,njk,nhk-> nih', systemOfCoordinatesRef,  E, systemOfCoordinatesRef)
    for i, e in enumerate(E_anatomic):
        if np.any(np.isnan(e)):
            E_anatomic[i] = np.eye(2)

    triangles= target.faces.reshape((-1, 4))[:, 1:]
    dihedralsRef = meshReconstruction.computeDihedralAngles( target.points,  target.cell_normals, edges)
    dihedrals0 = meshReconstruction.computeDihedralAngles(ref_undeformed.points, ref_undeformed.cell_normals, edges)
    dihedralsT = meshReconstruction.computeDihedralAngles(ref_deformed.points, ref_deformed.cell_normals, edges)

    pointsRef =  target.points[triangles]
    dihedrals_transformed = dihedralsRef + (dihedralsT - dihedrals0)
    coordinatesAnatomic = np.einsum('nij, nkj->nik', pointsRef, systemOfCoordinatesTarget)
    coordinatesAnatomic_transformed =np.einsum('nij, nkj->nki', E_anatomic, coordinatesAnatomic)
    triangleCoordinates3D = getCoordinatesTriangle(target, coordinatesAnatomic_transformed)
    if not linear:
        return deformMesh.deformMeshOptimisation(target, triangleCoordinates3D, dihedrals_transformed)
    else:
        return meshReconstruction.reconstructMeshLinear(triangleCoordinates3D, dihedrals_transformed, edges, triangles, target.GetNumberOfPoints())