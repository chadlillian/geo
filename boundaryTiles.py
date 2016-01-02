#!/home/chad/anaconda/bin/python

import  numpy as np
import  matplotlib.pyplot as plt
import  sys
import  urllib2
import  urllib
import  json
from    pymongo import MongoClient
from    shapely.geometry    import  Polygon
from    shapely.geometry    import  Point
import  zipfile
import  os
from    pykml  import  parser
import  kmlparse
import  sftest

class   boundaryTiles:
    def __init__(self):
        self.boundaries ={}
        self.meshTiles  =[]
        self.centroids  =[]
        self.boundariesSimp ={}

    def setPoints(self,place_name,points):
        self.boundaries = {}
        for i,section in enumerate(points):
            ces = zip(*section)
            polygon = Polygon(section)
            self.boundaries[i] = {'points':section,'coords':ces,'polygon':polygon}
        self.name = place_name
        print self.name
        
    def simplifyBoundaries(self,simp_eps,buf_eps):
        for bi in self.boundaries.keys():
            self.simplifyBoundary(bi,simp_eps,buf_eps)

    def simplifyBoundary(self,bi,simp_eps,buf_eps):
        polygon =self.boundaries[bi]['polygon']
        psimp   =polygon.simplify(simp_eps).buffer(buf_eps)
        self.boundaries[bi]['polygon_simple']   =psimp

    def getAllTiles(self,digits):
        N   =len(self.boundaries.keys())
        for k,bb in enumerate(self.boundaries.keys()):
            boundsize   =len(self.boundaries[bb]['points'])
            sys.stdout.flush()
            self.getTile(bb,digits)
        return  self.meshTiles

    def getBoundaryMinMax(self,bi,shape,digits):
        x,y =self.boundaries[bi][shape].exterior.coords.xy
        minlat  =np.round(min(x),digits)
        maxlat  =np.round(max(x),digits)
        minlon  =np.round(min(y),digits)
        maxlon  =np.round(max(y),digits)

        return  minlat,maxlat,minlon,maxlon
        
    def getTile(self,bb,digits):
        minlat,maxlat,minlon,maxlon =self.getBoundaryMinMax(bb,'polygon_simple',digits)

        res     =np.power(10.0,-digits)
        buf     =res*(0.5)
        lats    =np.arange(minlat,maxlat,res)
        lons    =np.arange(minlon,maxlon,res)

        mlats,mlons =np.meshgrid(lats,lons)
        mlen    =np.prod(mlats.shape)
        mlats   =np.reshape(mlats,(1,mlen))
        mlons   =np.reshape(mlons,(1,mlen))
    
        inside  =[]
        outside =[]
        frmt    =' '*20+'%f\r'
        for k,mt,mn in zip(range(mlen),mlats[0],mlons[0]):
            inout   =self.boundaries[bb]['polygon_simple'].contains(Point(mt,mn))
            if inout:
                inside.append([mt,mn])
            else:
                outside.append([mt,mn])
            sys.stdout.flush()
        
        if len(inside)==0:
            x,y =self.boundaries[bb]['polygon'].centroid.xy
            mt  =np.round(x[0],digits)
            mn  =np.round(y[0],digits)
            inside.append([mt,mn])

        for ins in inside:
            lowerleft   =[ins[0]-buf,ins[1]-buf]
            upperright  =[ins[0]+buf,ins[1]+buf]
            centroid    =[ins[0]    ,ins[1]    ]
            tile    ={'lowerleft':lowerleft, 'upperright':upperright,'centroid':centroid}
            if centroid not in self.centroids:  #   avoid repeating tiles
                self.centroids.append(centroid)
                self.meshTiles.append(tile)
            
        return  {'inside':zip(*inside),'outside':zip(*outside)}
    
    def plotTiles(self):
        for bi in self.boundaries:
            b   =self.boundaries[bi]
            plt.plot(b['coords'][0],b['coords'][1],linewidth=3,color='k')

            b   =self.boundaries[bi]['polygon_simple']
            x,y =b.exterior.coords.xy
            plt.plot(x,y,linewidth=1,color='r',alpha=0.3)
        for t in self.meshTiles:
            ll  =t['lowerleft']
            ur  =t['upperright']
            eps =(ur[0]-ll[0])*0.00
            square0 =[ll[0]+eps,ll[0]+eps,ur[0]-eps,ur[0]-eps,ll[0]+eps]
            square1 =[ll[1]+eps,ur[1]-eps,ur[1]-eps,ll[1]+eps,ll[1]+eps]
            plt.plot(square0,square1,'b')
        plt.show()

    def getTiles(self):
        return  self.meshTiles
    
    def getName(self):
        return  self.name
        
if __name__=="__main__":
    file_name =sys.argv[1]
    place_name = sys.argv[2]

    b = sftest.shapeFileReader()
    d = b.open(file_name,place_name)

    res =1
    eps_simp    =np.power(10.0,-res)

    c = boundaryTiles()
    c.setPoints(place_name,d)
    c.simplifyBoundaries(eps_simp,1.0*eps_simp)
    c.getAllTiles(res)
    c.plotTiles()

