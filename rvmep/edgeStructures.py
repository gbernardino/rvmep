import collections, numpy as np

class EdgeLocator:
    """
    Dictionary to obtain all faces incident in an edge. Returns the index.
    Warning: assumes an orientable triangular mesh. IE, only at most 2 faces share the same edge, and the are traversed in each direction.
    """ 
    def add_edge(self, a, b, fId):
        if (a,b) in self.edgeDictionary and self.manifold:
            raise Exception('The  mesh is non manifold! Two edges in the same order.')
        elif (a,b) in self.edgeDictionary and not self.manifold:
            self.edgeDictionary[(b,a)] = fId
        else:
            self.edgeDictionary[(a,b)] = fId
 
    def buildLocator(self, vtkMesh, manifold = True):
        self.manifold = manifold
        self.edgeDictionary = collections.defaultdict(lambda: None)
        for i in range(vtkMesh.GetNumberOfCells()):
            p = [int(vtkMesh.GetCell(i).GetPointId(j)) for j in range(3)]
            self.add_edge(p[0], p[1], i)
            self.add_edge(p[1], p[2], i)
            self.add_edge(p[2], p[0], i)

    def buildLocatorFromNP(self, faces, manifold = True):
        self.manifold = manifold
        self.edgeDictionary = collections.defaultdict(lambda: None)
        for i,t  in enumerate(faces):
            self.add_edge(t[0], t[1], i)
            self.add_edge(t[1], t[2], i)
            self.add_edge(t[2], t[0], i)

    def __call__(self, edgeStart, edgeEnd):
        return self.locateFaceByEdge(edgeStart, edgeEnd)
    
    def locateFaceByEdge(self, edgeStart, edgeEnd):
        """
        Returns the index of the face incident with that edge, traversed in the specified direction
        """
        
        return self.edgeDictionary[(edgeStart, edgeEnd)]

    def locateAllFacesByEdge(self, edgeStart, edgeEnd):
        """
        Returns a list withthe index of all faces incident with that edge (ie 2)
        """
        
        return list(filter(lambda s: s is not None, [self.edgeDictionary[(edgeStart, edgeEnd)], self.edgeDictionary[(edgeEnd, edgeStart)] ]))

    def getEdges(self, bothEnds = False):
        if bothEnds:
            return set(list(self.edgeDictionary.keys()) + [(a, b) for b, a in self.edgeDictionary.keys()])
        else:
            return set(self.edgeDictionary.keys())
class EdgeFace:
    def __init__(self, fId, t, e):
        self.fId = fId
        self.pBegin = np.where(t == e[0])[0][0]
        self.pEnd = np.where(t == e[1])[0][0]
        self.pOther = 3 - self.pEnd - self.pBegin

def getEdges(vtkMesh):
    """
    Gets a list 
    """
    locator = EdgeLocator()
    locator.buildLocator(vtkMesh)
    edgeFaces = collections.namedtuple('EdgeFaces', 'edge f1 f2')
    triangles = vtkMesh.faces.reshape((-1, 4))[:, 1:]
    
    _edgesNonDegenerate = []
    edges = locator.getEdges(False)
    for e in edges:
        if e[0] < e[1] or (e[1], e[0]) not in edges:
            continue
        f = locator.locateAllFacesByEdge(e[0], e[1])
        _edgesNonDegenerate.append(edgeFaces(e, 
                                                    EdgeFace(f[0], triangles[f[0]], e),
                                                    EdgeFace(f[1], triangles[f[1]], e)))
    return _edgesNonDegenerate

def getEdgesAndNeighbours(m, edgesNonDegenerate = None):
    """
    Gets a list of, for each edge, which are the incident faces.
    """
    if edgesNonDegenerate is None:
        edgesNonDegenerate = getEdges(m)
    nEdges = len(edgesNonDegenerate)
    edgesNP = np.zeros((nEdges, 2), dtype = np.uint32)
    neighboursDirect = collections.defaultdict(list)  #Which edges ae 
    neighboursInDirect = collections.defaultdict(list)

    for i, e in enumerate(edgesNonDegenerate):
        edgesNP[i,0] = e.f1.fId
        edgesNP[i,1] = e.f2.fId
        neighboursDirect[e.f2.fId].append(i)   # Which edges are incident to a specific face
        neighboursInDirect[e.f1.fId].append(i) # Which edges are incident to a specific face

    neighboursNP = np.zeros([m.GetNumberOfCells(), 3], dtype = np.uint32)
    for i, _ in enumerate(neighboursNP):
        j = 0
        for k in neighboursDirect[i]:
            neighboursNP[i, j] = k
            j += 1

        for k in neighboursInDirect[i]:
            neighboursNP[i, j] = k + nEdges
            j += 1
            
        while j < 3:
            """
            Make sure that there are always 3 neighbours, even for triangles in the border. For those, copy the last edge.
            """
            neighboursNP[i,j] = neighboursNP[i, j-1]
            j += 1
    return edgesNP, neighboursNP
