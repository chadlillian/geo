#!/home/chad/anaconda/bin/python

import  os
import  sys
import  zipfile
import  codecs
import  xml.etree.ElementTree as ET
import  shapefile
import  StringIO

sys.stdout  =codecs.getwriter('utf-8')(sys.stdout)

# class to read in a shapefile (ESRI) and find the geometry of a particular region
class shapeFileReader:

    # open a shapefile.zip file find the files in it and look for
    #   an entity called place_name
    def open(self,filename,place_name):
        self.zip_file_name = filename
        self.readFileNames()
        self.findRegion(place_name)
    
    def getName(self):
        return self.db_name

    def getBoundaries(self):
        return self.boundaries

    def display(self,filename):
        self.zip_file_name = filename
        self.readFileNames()
        self.findRegion("")

    # open up the zip file and put contents into a dict based on level
    def readFileNames(self):
        self.zsf = zipfile.ZipFile(self.zip_file_name)
        zsfs = self.zsf.namelist()

        self.shape_file_names = {}
        for s in zsfs:
            num = s.split('.')[0][-1]   # gets the level number (assumed <10)
            if num.isdigit():
                num = int(num)
                ext = s.split('.')[1]
                if num in self.shape_file_names.keys():
                    self.shape_file_names[num][ext] = s
                else:
                    self.shape_file_names[num] = {ext:s}

    
    # open each of the shapefiles (starting at coarsest level 0)
    #   search each one for the region_name of interest then return its geometry
    #   in the case of regions with the same name at different levels of coarseness
    #   the coarsest region is returned
    def findRegion(self,region_name):
        for i in range(len(self.shape_file_names.keys())):
            # treat the string as a file is the only way to get shapefile.Reader to work on a zip file
            zshp = StringIO.StringIO(self.zsf.open(self.shape_file_names[i]['shp']).read())
            zdbf = StringIO.StringIO(self.zsf.open(self.shape_file_names[i]['dbf']).read())
            sf = shapefile.Reader(shp=zshp,dbf=zdbf)
    
            shapes = sf.shapes()
            records = sf.records()
            fieldsindices = [j-1 for j,f in enumerate(sf.fields) if f[0].lower().find('name_')==0 and f[0].split('_')[1].isdigit()]
            for shape,record in zip(shapes,records):
                names = [record[j].decode('utf-8') for j in fieldsindices]
                keyname = names[-1]#.decode('utf-8')
                print keyname
                if keyname == region_name:
                    parts = list(shape.parts) + [len(shape.points)]
                    self.boundaries = [shape.points[parts[i]:parts[i+1]] for i in range(len(parts)-1)]
                    self.db_name = '_'.join(names)
                    return
    
    #   returns a list of lists of coordinates
    def getGeometry(self,place_name):
        print self.regions[place_name]

if __name__=="__main__":
    a = shapeFileReader()
    #b = a.open(sys.argv[1],sys.argv[2])
    a.display(sys.argv[1])
    #a.getGeometry(sys.argv[2])
