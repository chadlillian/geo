#!/home/chad/anaconda/bin/python
from geojson import MultiPolygon, Feature, Polygon,MultiPoint, Point
import geojson
import numpy as np

class   gjson:
    def __init__(self):
        self.points_used = {}

    def initFile(self,outfilename):
        self.outfile = open(outfilename,'w')

    def closeFile(self):
        self.outfile.close()
        
    def rgb2hex(self,rgb):
        red = hex(int(float(rgb[0]*255)))+'0'
        grn = hex(int(float(rgb[1]*255)))+'0'
        blu = hex(int(float(rgb[2]*255)))+'0'
        color = "#"+red[2:4]+grn[2:4]+blu[2:4]

        return color

    def writePointsFeatureCollection(self,latlons,colors,varname):
        featureCollection = {'features':[]}
        print latlons.shape,colors.shape
        for latlon,color in zip(latlons,colors):
            hexcolor = self.rgb2hex(color)
            st = {"fillcolor": hexcolor, "radius":40}
            #st = {"fillOpacity":0.3, "fillcolor": hexcolor, "color":hexcolor,"radius":40, "stroke":False}
            pr = {"style":st}
            ft = Feature(geometry=Point(latlon.tolist()), properties=pr)
            featureCollection['features'].append(ft)

        print>>self.outfile, "var %s = "%(varname)
        geojson.dump(featureCollection,self.outfile)

    def writeMultiPolygonFeatureCollection(self,polygons,colors,varname):
        featureCollection = {'features':[]}
        for polygon,color in zip(polygons,colors):
            st = {"fillcolor": color, "color":color}
            pr = {"style":st}
            ft = Feature(geometry=Polygon(polygon), properties=pr)
            featureCollection['features'].append(ft)

        print>>self.outfile, "var %s = "%(varname)
        geojson.dump(featureCollection,self.outfile)


    def writeMultiPolygonFeatureCollection_(self,polygons,**kwargs):
        featureCollection = {'features':[]}
        options = kwargs.keys()
        vals = [kwargs[o] for o in options]
        styles = [dict(zip(*(options,val))) for val in zip(*vals)]

        for polygon,st in zip(polygons,styles):
            pr = {"style":st}
            ft = Feature(geometry=Polygon(polygon), properties=pr)
            featureCollection['features'].append(ft)

        print>>self.outfile, "var data = "
        geojson.dump(featureCollection,self.outfile)


