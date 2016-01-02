#!/home/chad/anaconda/bin/python
# http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

from shapely.ops import cascaded_union, polygonize
import shapely.geometry as geometry
from scipy.spatial import Delaunay
import numpy as np
import math
import matplotlib.pyplot as plt
 
def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
 
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
 
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
                # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
 
    #coords = np.array([point.coords[0] for point in points])
    coords = points
 
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
 
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
 
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
 
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
 
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
 
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    #return cascaded_union(triangles), edge_points
    return cascaded_union(triangles)
 

def alpha_shape_wrapper(points,alpha):
    #concave_hull, edge_points = alpha_shape(points,alpha)
    concave_hull = alpha_shape(points,alpha)
    if concave_hull.type=='MultiPolygon':
        xy = concave_hull[0].exterior.coords.xy
    else:        
        xy = concave_hull.exterior.coords.xy
    
    ret = [[x,y] for x,y in zip(xy[0],xy[1])]
    return ret
    

if __name__=="__main__":
    N = 1000
    m = 10
    alpha = 5.0
    alpha = 1.0/alpha
    r = np.random.randn(N,2)
    #c = np.random.rand(N,1)
    #cx = m*np.sin(5.54*c)
    #cy = m*np.cos(5.54*c)
    c = np.random.randn(N,1)
    cx = m*np.sin(0.8*c)
    cy = m*np.cos(0.8*c)
    points = r+np.hstack((cx,cy))
    
    print points
    xy = alpha_shape_wrapper(points,alpha)
    
    xy = zip(*xy)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points[:,0],points[:,1])
    ax.plot(xy[0],xy[1],color='red')
    plt.show()

