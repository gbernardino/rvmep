import sfepy, vtk, pyvista, tetgen, gdist

from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete import (FieldVariable, Material, Integral, Function, Equation, Equations, Problem)
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.terms import Term
from sfepy.terms.terms_diffusion import LaplaceTerm
import numpy as np, collections
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from . import tools

def doPartitionSequence(meshes, pointsPulmonary, pointsTricuspid, apexId):
    meshPartitionSurfaceGeodesics(meshes[0], pointsPulmonary, pointsTricuspid, apexId)
    for m in meshes[1:]:
        meshPartitionSurfacePropagate(m, meshes[0])
    results = []
    for m in meshes:
        results.append(doPartition(m))
    return results

ResultsParcellation = collections.namedtuple('ResultsParcellation', 'apical inlet rvot')
def doPartition(mesh):
    tet = tetgen.TetGen(mesh)
    tet.tetrahedralize(minratio=1.5)
    mesh3D = tet.grid
    
    distancePulmonary = solveLaplaceEquationTetrahedral(mesh3D, mesh, 'distancePulmonary')
    distanceApex = solveLaplaceEquationTetrahedral(mesh3D, mesh, 'distanceApex')
    distanceTricuspid = solveLaplaceEquationTetrahedral(mesh3D, mesh, 'distanceTricuspid')

    tools.addArrayToMeshVTK(mesh3D, np.minimum(distanceApex, distanceTricuspid) - distancePulmonary, 'pv')
    tools.addArrayToMeshVTK(mesh3D, distanceTricuspid  - distanceApex, 'pa')
    tools.addArrayToMeshVTK(mesh3D, distanceApex  - distanceTricuspid, 'pt')

    inletRVOT, apex =  splitVolumes(mesh3D, 0, scalarFieldName= 'pa')
    rvot, inlet =  splitVolumes(inletRVOT, 0, scalarFieldName= 'pv')
    return ResultsParcellation(apex, rvot, inlet)

def splitVolumes(meshVTK, th, scalarFieldName):
    """
    splits the volume using the field "scalarFieldName" and the threshold value "th"
    """
    meshVTK.GetPointData().SetActiveScalars(scalarFieldName)
    
    clip = vtk.vtkClipDataSet()
    clip.SetInputData(meshVTK)
    clip.SetValue(th)
    clip.SetInsideOut(True) # Get <= 
    clip.Update()
    
    tetrahedrilize = vtk.vtkDataSetTriangleFilter()
    tetrahedrilize.SetInputConnection(clip.GetOutputPort())
    tetrahedrilize.Update()
    part1 = tetrahedrilize.GetOutput()
    
    clip2 = vtk.vtkClipDataSet()
    clip2.SetInputData(meshVTK)
    clip2.SetValue(th)
    clip2.SetInsideOut(False)
    clip2.Update()
    
    tetrahedrilize2 = vtk.vtkDataSetTriangleFilter()
    tetrahedrilize2.SetInputConnection(clip2.GetOutputPort())
    tetrahedrilize2.Update()
    part2 = tetrahedrilize2.GetOutput()
    
    return pyvista.UnstructuredGrid(part1), pyvista.UnstructuredGrid(part2)

def meshPartitionSurfaceGeodesics(mesh, pointsPulmonary, pointsTricuspid, apexId):
    """
    Does the partition    
    """

    points, faces = tools.pyvistaToNumpy(mesh)
    faces = faces.astype(np.int32)
    distancePulmonary = gdist.compute_gdist(points, faces, pointsPulmonary.astype(np.int32))
    distanceTricuspid = gdist.compute_gdist(points, faces, pointsTricuspid.astype(np.int32))
    distanceApex = gdist.compute_gdist(points, faces, np.array([apexId], dtype = np.int32))

    tools.addArrayToMeshVTK(mesh,distancePulmonary, 'distancePulmonary')
    tools.addArrayToMeshVTK(mesh,distanceApex, 'distanceApex')
    tools.addArrayToMeshVTK(mesh,distanceTricuspid, 'distanceTricuspid')
    return mesh

def meshPartitionSurfacePropagate(mesh, meshRef):
    """
    Propagates a partition defined over a surface
    """
    for m in ['distancePulmonary', 'distanceTricuspid', 'distanceApex']:
        mesh.GetPointData().AddArray(meshRef.GetPointData().GetArray(m))

class ClosestPoint:
    """
    Trivial solution of finding the closest point in a set by simply computing all distances
    """
    def __init__(self, points, val, vtkMesh):
        self.points = points
        self.val = val
        self.j = 0
        self.locator = vtk.vtkCellLocator()
        self.locator.SetDataSet(vtkMesh)
        self.locator.BuildLocator()
        
    def findClosestPoint(self, p):
        # TODO: compute the closest point, compute the triangle coordinates of the intersection point in that triangle, and
        subId = vtk.mutable(0) 
        meshPoint = np.zeros(3)
        cellId = vtk.mutable(0) 
        dist2 =  vtk.mutable(0.) 
        self.locator.FindClosestPoint(p, meshPoint, cellId, subId, dist2)
        meshPoint
        cellId
        np.linalg.solve(self.points[self.triangles[cellId]], meshPoint)
    
    def interpolate(self, coors):
        """
        Define in base of coordinates...
        See if interpolations are possible
        """
        i = np.argmin(np.linalg.norm(coors - self.points, axis = 1))
        return self.val[i]

def solveLaplaceEquationTetrahedral(mesh3D, surfaceMeshVTK, nameArray):
    """
    Does Laplace interpolation from the surface mesh to the interior

    mesh: path to a 3D mesh / numpy mesh
    
    Warning: it does quite a bit of recomputation, it is far from optimal, but should be fast enough. The domains and stiffness matrix can be reused when interpolating different fields on the same mesh.
    """
    if isinstance(mesh3D, str):
        mesh3D = Mesh.from_file(mesh3D)
    else:
        #If it is from numpy
        mesh3D = sfepy.discrete.fem.Mesh.from_data('mesh3d', mesh3D.points, None,
                    [mesh3D.cells.reshape((-1, 5))[:, 1:] , ], [np.zeros(mesh3D.n_cells, dtype=np.int32)], ['3_4'])
    #Set domains
    domain = FEDomain('domain', mesh3D)
    omega = domain.create_region('Omega', 'all')
    boundary = domain.create_region('gamma', 'vertex  %s' % ','.join(map(str, range(surfaceMeshVTK.GetNumberOfPoints()))), 'facet')

    #set fields
    field = Field.from_args('fu', np.float64, 1, omega, approx_order=1)
    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')
    m = Material('m', val = [1.])

    #Define element integrals
    integral = Integral('i', order=3)

    #Equations defining 
    t1 = Term.new('dw_laplace( v, u )',
            integral, omega,v=v, u=u)
    eq = Equation('balance', t1)
    eqs = Equations([eq])
    
    
    heatBoundary = vtk_to_numpy(surfaceMeshVTK.GetPointData().GetArray(nameArray))
    points = surfaceMeshVTK.points

    #Boundary conditions
    c = ClosestPoint(points,heatBoundary, surfaceMeshVTK)

    def u_fun(ts, coors, bc=None, problem=None, c = c):
        c.distances = []
        v = np.zeros(len(coors))
        for i, p in enumerate(coors):
            v[i] = c.interpolate(p)
            #c.findClosestPoint(p)
        return v

    bc_fun = Function('u_fun', u_fun)
    fix1 = EssentialBC('fix_u', boundary, {'u.all' : bc_fun})
    
    #Solve problem
    ls = ScipyDirect({})
    nls = Newton({}, lin_solver=ls)

    pb = Problem('heat', equations=eqs)
    pb.set_bcs(ebcs=Conditions([fix1]))

    pb.set_solver(nls)
    state = pb.solve(verbose = False, save_results = False)
    u = state.get_parts()['u']
    return u