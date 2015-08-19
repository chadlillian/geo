#!/home/chad/anaconda/bin/python

import	os
import	sys
import	zipfile
from pykml	import	parser
import	codecs

sys.stdout	=codecs.getwriter('utf-8')(sys.stdout)

class	kmlparse:
	def	__init__(self):
		self.boundaries	={}

	def	readBoundaryFile(self):
		placemarks	=[p for p in self.doc.Document.Folder.Placemark]
		for pm in placemarks:
			attribs	=dict([(a.attrib['name'],a.text) for a in pm.ExtendedData.SchemaData.SimpleData[0]])
			depth	=len([xx for xx in attribs.keys() if xx.find('ID_')==0])
			coords	=pm.MultiGeometry.Polygon.outerBoundaryIs.LinearRing.coordinates

			coords	=[b.LinearRing.coordinates for b in pm.MultiGeometry.Polygon.getchildren()]
			coords	=[[map(float,x.split(',')) for x in b.LinearRing.coordinates.text.split()] for b in pm.MultiGeometry.Polygon.getchildren()]
			names	=[attribs['NAME_%i'%(i)] for i in range(depth)]
			ret	=attribs
			ret['coords']	=coords
			id_		='ID_%i'%(depth-1)
			self.boundaries[attribs[id_]]	=ret
	
	def	openKML(self,kmlfile):
		if os.path.splitext(kmlfile)[1]=='.kmz':
			kmlfile		=zipfile.ZipFile(kmlfile)
			kmlfilename	=kmlfile.namelist()[0]
			coordfile	=kmlfile.read(kmlfilename)
		else:
			coordfile	=open(kmlfile).read()

		self.kmlfile	=coordfile
		self.doc		=parser.fromstring(coordfile)
	
	def	scanKML(self):
		for b in self.boundaries.keys():
			print b,len(self.boundaries[b])
	
	def	queryBoundaries(self,qry):
		ret	=False
		for b in self.boundaries.keys():
			if b==qry:
				ret	=self.boundaries[b]

		self.getPlaceName(ret)
		ret['placename']	=self.placename
		return	ret
	
	def	queryKML(self,kmlfilename,qry):
		self.openKML(kmlfilename)
		self.readBoundaryFile()
		a	=self.queryBoundaries(qry)
		return	a
	
	def	getPlaceName(self,b):
		n	=[nn for nn in b.keys() if nn.find('NAME')==0]
		n.sort()
		self.placename	='_'.join([b[nn] for nn in n])
		
	
if __name__=="__main__":
	kmlfile	=sys.argv[1]
	qry		=sys.argv[2]
	
	a	=kmlparse()
	a.openKML(kmlfile)
	a.readBoundaryFile()
	b	=a.queryBoundaries(qry)
	print b['placename']


