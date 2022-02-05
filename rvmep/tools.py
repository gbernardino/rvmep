import numpy as np, pyvista
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def read_ucd_to_vtk(path):
    """
    Reads an ucd file (the format in which Tomtec meshes are exported) and transforms it to a pyvista object.
    A part of the format is hardcoded (the amount of slacklines after reading the size of the object is fixed), so make sure it is consistent
    """
    with open(path, mode = 'r') as f:
        for i, l in enumerate(f.readlines()):
            if i == 0:
                nPoints, nFaces, _, _, _ = list(map(int, l.split()))
                points = np.zeros([nPoints, 3])
                triangles = np.zeros([nFaces, 3], dtype = np.uint16)
                pointData = np.zeros([nPoints, 2])

                iFaces = 0
                iPoints = 0
                iPointsData = 0
                slackLines = 3  # 3 slack lines that are not readed

            elif iPoints < nPoints:
                points[iPoints] = list(map(float, l.split()[1:]))
                iPoints += 1

            elif iFaces < nFaces:
                triangles[iFaces] = list(map(int, l.split()[-3 : ]))
                iFaces += 1

            elif slackLines:
                slackLines -= 1

            elif iPointsData < nPoints:
                pointData[iPointsData] = list(map(float, l.split()[-2 : ]))
                iPointsData += 1
    return numpyToPyvista(points, triangles)

def getTomtecApexValvePointsRV():
    """
    Hand selected points from the TOMTEC RV model
    """
    pointsTricuspid = np.array([388, 389, 392, 393, 144, 540, 145, 538, 539, 422, 423, 38, 541, 49, 55, 328, 329, 332, 333, 87, 94, 100, 101, 103, 104, 105, 122, 123, 126, 127])
    pointsPulmonary = np.array([410, 411, 409, 408, 53, 64, 65, 66, 67, 68, 69, 83, 476, 477, 92, 478, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 479])
    apexId = 906
    return apexId, pointsTricuspid, pointsPulmonary

def numpyToPyvista(points, triangles):
    return pyvista.PolyData(points,  np.concatenate((3*np.ones([len(triangles), 1], dtype = np.uint16) , triangles), axis = 1).flatten())

def pyvistaToNumpy(mesh):
    return mesh.points, mesh.faces.reshape((-1, 4))[:, 1:]

def addArrayToMeshVTK(mesh, array, name, domain = 'points'):
    arrayVTK = numpy_to_vtk(array)
    arrayVTK.SetName(name)

    if domain == 'points':
         mesh.GetPointData().AddArray(arrayVTK)
    elif domain == 'faces':
         mesh.GetPointData().AddArray(arrayVTK)
    else:
        raise ValueError('Unknown domain')