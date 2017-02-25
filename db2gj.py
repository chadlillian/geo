#!/home/chad/anaconda/bin/python
from    scipy   import  stats
from    scipy.spatial import ConvexHull
from    mpl_toolkits.basemap import Basemap
import  matplotlib.pyplot as plt
import  matplotlib.figure as pltfig
from    mpl_toolkits.mplot3d   import Axes3D
import  numpy as np
import  sys
from    pymongo import MongoClient
import  math
import  pylab   as P
from    sklearn.cluster import DBSCAN
from    geojson import MultiPolygon, Feature, Polygon,MultiPoint, Point
import  geojson
from	sklearn.decomposition	import	PCA
from    sklearn import preprocessing
from    sklearn.neighbors import NearestNeighbors
from    sklearn.neighbors.kde import KernelDensity
from    collections import Counter
import  alphashape
import  gjson as gj
import  optics as op

class   plotstuff:
    def __init__(self):
        self.client =MongoClient()
        self.points_used = {}

    def setupDB(self,db,cols):
        self.db     =self.client[db]
        self.cols = [self.db[col] for col in cols]

    def getCollectionStatistics(self,samples):
        neigh =  NearestNeighbors(n_neighbors=samples)
        neigh.fit(self.coordinates)
        A = neigh.kneighbors_graph(self.coordinates,mode='distance')
        b = A.nonzero()
        c = np.log10(np.array(A[b[0],b[1]]))

        mean = c[0].mean()
        std = c[0].std()
        pc = np.percentile(c[0],50)

        n,bins,patches = plt.hist(c[0],80)
        plt.show()
        mx = bins[n.argmax()]

        self.collection_stats = {'mean':np.power(10,mean),'std':np.power(10,std),'pcntl':np.power(10,pc), 'max':np.power(10,mx)}
        return self.collection_stats
        
    # get all coordinates, predictions, and photo urls from the pre-selected collection
    def getAllData(self,catfile):
        categories = [line.split()[0] for line in open(catfile).readlines()]
        data = []
        coords = []
        photos = []
        for col in self.cols:
            docs = col.find({"$and":[{'prediction':{"$ne": 0}},{'prediction': {"$exists":True}}]},timeout=False)

            for doc in docs:
                if doc['prediction']:
                    prediction = [doc['prediction'][category] for category in categories]
                    coordinates = [doc['latitude'],doc['longitude']]
                    photo = doc['photo_file_url']
                    
                    data.append(prediction)
                    coords.append(coordinates)
                    photos.append(photo)

        npdata = np.asarray(data)
        npcoords = np.asarray(coords)

        self.predictions = npdata
        self.coordinates = npcoords
        self.photos = photos

    def arrayToRGB(self,a):
        ms = preprocessing.MinMaxScaler()
        fc = ms.fit_transform(a)
        hxpower = np.matrix([256**3,256**2,256**1]).T
        hxcolor = np.dot(fc,hxpower)
        hxcolor = map(hex,map(int,np.array(np.dot(fc,hxpower).T).tolist()[0]))
        hxcolor = [ '#'+('000000'+hxc[2:])[-6:] for hxc in hxcolor]
        return hxcolor

    def getDistinctRGB(self,N):
        n = int(np.ceil(np.power(N,1.0/2.0)))
        z = 0
        ret = {}
        for i in range(n):
            for j in range(n):
                red = hex(int(255*float(i)/n))+"00"
                green = hex(int(255))+"00"
                blue = hex(int(255*float(j)/n)) +"00"
                ret[z] = '#'+red[2:4]+green[2:4]+blue[2:4]
                z = z+1
        return ret
                    
    def clusterCoordinates(self,eps,min_samples):
        r = self.dbscan(self.coordinates,self.predictions,eps,min_samples)
        rks = r['data_active'].keys()
        rks.sort()
        x = []
        y = []
        c = []
        s = []
        
        predictions = []
        cluster_radii = []
        polygons = []

        print "number of clusters = ",len(rks)
        for cl in rks[1:]:
            coords = np.array(r['data_active'][cl])
            preds = np.array(r['data_inert'][cl])

            predictions.append(np.max(preds,0))
            center = np.mean(coords,0)
            radius = np.std(coords,0)
            cluster_radii.append(radius)
            skr = zip(*r['data_active'][cl])
            color = [cl+1]*len(skr[0])
            size = [1+400*(cl>=0)]*len(skr[0])

            x.extend(skr[1])
            y.extend(skr[0])
            c.extend(color)
            s.extend(size)

            try:
                polygon = [alphashape.alpha_shape_wrapper(coords[:,(1,0)],50.0),]
            except:
                n = 9
                polygon = [[[center[1]+radius[1]*np.cos(i*3.1415/n),center[0]+radius[0]*np.sin(i*3.1415/n)] for i in range(n)]]
            polygons.append(polygon)

        nppreds = np.log10(np.asarray(predictions))
        pcapred = self.pca(nppreds,6)
        ms = preprocessing.MinMaxScaler()
        fc = ms.fit_transform(pcapred)

        # find low probability points and color them opaque
        cols = (0,1,2,3,4,5)
        kde = KernelDensity(kernel='gaussian',bandwidth=0.05).fit(fc[:,cols])
        kdescores = kde.score_samples(fc[:,cols])
        ss = preprocessing.MinMaxScaler()

        datan = np.power(10.0,(0.01*ss.fit_transform(kdescores)))
        opacity = ss.fit_transform(datan).tolist()
        zero = (1.0+0.0*datan).tolist()
        plt.hist(opacity,50)
        plt.show()

        hxcolor1 = ["#%02x%02x%02x"%tuple([255*aa for aa in x][0:3]) for x in fc.tolist()]
        hxcolor2 = ["#%02x%02x%02x"%tuple([255*aa for aa in x][3:6]) for x in fc.tolist()]

        C = np.array([x,y]).T
        color = np.array(c)
        npcr = np.asarray(cluster_radii)
        n = 17
        self.makePointsToMultiPointFeatureCollection(self.coordinates[::n,(1,0)],self.coordinates[::n,0],self.coordinates[::n,1],'sample-geojson-cluster2.js')

        g = gj.gjson()
        g.initFile('sample-geojson-poly.js')
        #g.writeMultiPolygonFeatureCollection_(polygons,fillColor=hxcolor1,fillOpacity=opacity,color=hxcolor2,opacity=opacity)
        g.writeMultiPolygonFeatureCollection_(polygons,fillColor=hxcolor1,color=hxcolor2,fillOpacity=opacity)
        g.closeFile()

    def getCategoryCutoff(self,category,cutoff):
        #docs    =self.col.find({'prediction'+'.'+category :{"$gt": cutoff}})
        rtn = []
        ids = set() # to avoid inserting duplicate photos, when collection boundaries overlap
        for col in self.cols:
            docs = col.find({'prediction'+'.'+category :{"$gt": cutoff}})
            for doc in docs:
                rt  ={'photo_id':doc['photo_id'], 'longitude':doc['longitude'],'latitude':doc['latitude'],category:doc['prediction'][category]}
                if doc['photo_id'] not in ids:
                    rtn.append(rt)
                    ids.add(doc['photo_id'])
        return  rtn

    # given a category this finds clusters using the DBSCAN algorithm
    # returns a dictionary with 
    #   index:cluster number
    #   value:list of dictionares (dictionaries given by getCategoryCutoff)
    def getCategoryClusters(self,category,eps,min_samples,cutoff):
        data = self.getCategoryCutoff(category,cutoff)
        self.category = category

        lons        =[ a['longitude'] for a in data]
        lats        =[ a['latitude']  for a in data]

        ll          =np.array([[n,t] for n,t in zip(lons,lats)])
        #dd = self.getNeighborStatistics(ll,min_samples*2,70)
        db          =DBSCAN(eps=eps, min_samples=min_samples).fit(ll)
        clusterid   =db.labels_
        clusters    =list(set(clusterid))
        ret         =dict([(cl,[]) for cl in clusters])

        for cl,d in zip(clusterid,data):
            ret[cl].append(d)

        self.clusters = ret

    # make convex hulls for each of the clusters
    def getCategoryPolygons(self):
        catClust = self.clusters
        poly    =[]
        for k in catClust.keys():
            points  =np.array([[a['longitude'],a['latitude']] for a in catClust[k]])
            ids = [a['photo_id'] for a in catClust[k]]
            if k>=0 and points.shape[0]>2:
                hull    =ConvexHull(points)
                hullvert    =list(hull.vertices)
                coords  =[(x,y) for x,y in points[hullvert,:]]
#                coords = alphashape.alpha_shape_wrapper(points,.01)
                poly.append((coords,))

                for p,i in zip(points,ids):
                    if i not in self.points_used.keys():
                        self.points_used[i] = {'coordinates':p.tolist(), 'photo_id':i, 'categories':[self.category]}
                    else:
                        self.points_used[i]['categories'].append(self.category)

        mp = MultiPolygon(poly)
        self.category_multipolygon = mp

    def writeFeature(self,style,properties):
        ft = Feature(geometry=self.category_multipolygon, properties=pr)

    def getNeighborStatistics(self,data,samples,pcntl):
        neigh =  NearestNeighbors(n_neighbors=samples)
        neigh.fit(data)
        A = neigh.kneighbors_graph(data,mode='distance')
        b = A.nonzero()
        c = np.log10(np.array(A[b[0],b[1]]))
        mean = c[0].mean()
        std = c[0].std()
        pc = np.percentile(c[0],pcntl)

        n,bins,patches = plt.hist(c[0],50)
        plt.show()
        mx = bins[n.argmax()]
        ret = {'mean':np.power(10,mean),'std':np.power(10,std),'pcntl':np.power(10,pc), 'max':np.power(10,mx)}

        return ret
        
    def makePointsToMultiPointFeatureCollection(self,points,colors,radii,outfilename):
        featureCollection = {'features':[]}
        for color,radius,point in zip(colors,radii,points):
            st = {"fillOpacity":1,"fillcolor": color, "color":color,"radius":radius,"stroke":False}
            pr = {"style":st}
            ft = Feature(geometry=Point(point.tolist()),properties=pr)
            featureCollection['features'].append(ft)
        outfile = open(outfilename,'w')
        print>>outfile, "var data = "
        geojson.dump(featureCollection,outfile)
        outfile.close()
        
    def makeMultiPointFeatureCollection(self,category,cutoff,eps,min_samples,outfilename):
        self.featureCollection = {'features':[]}
        data = self.getCategoryCutoff(category,cutoff)
        pointmatrix = []
        for datum in data:
            intensity = datum[category]
            red = hex(int(float(intensity*255)))+'0'
            green = "00"
            blue = hex(int(float((1.0-intensity)*255)))+'0'
            color = "#"+red[2:4]+green+blue[2:4]
            st = {"fillcolor": color, "color":color}
            pr = {"name":category, "popupContent":category+'\n%f'%(intensity),"style":st}
            ft = Feature(geometry=Point([datum['longitude'],datum['latitude']]), properties=pr)
            pointmatrix.append([datum['longitude'],datum['latitude']])
            self.featureCollection['features'].append(ft)
        outfile = open(outfilename,'w')
        print>>outfile, "var data = "
        geojson.dump(self.featureCollection,outfile)
        outfile.close()

    def makeFeatureCollection(self,categories,cutoff,eps,min_samples,outfilename):
        self.featureCollection = {'features':[]}
        n = len(categories)-1
        for i,category in enumerate(categories):
            print category
            self.getCategoryClusters(category,eps,min_samples,cutoff)
            self.getCategoryPolygons()
            red = hex(int(float(i*255)/n))+'0'
            green = "00"
            blue = hex(int(float((n-i)*255)/n))+'0'
            color = "#"+red[2:4]+green+blue[2:4]
            st = { "weight": 2, "color": "#999", "opacity": 0, "fillColor": color, "fillOpacity": 0.3 }
            st = { "weight": 5, "color": color, "opacity": 0.5, "fillColor": color, "fillOpacity": 0.0 }
            pr = {"name":category, "popupContent":category,"style":st}
            ft = Feature(geometry=self.category_multipolygon, properties=pr)
            self.featureCollection['features'].append(ft)

        outfile = open(outfilename,'w')
        print>>outfile, "var data = "
        geojson.dump(self.featureCollection,outfile)
        #geojson.dump(ft,outfile)
        outfile.close()

    def pca(self,data,dim):
        pca_matrix = np.asarray(data)
        pca_ = PCA(n_components=dim)
        pca_.fit(pca_matrix)
        b = pca_.transform(pca_matrix)

        return b

    def optics(self,data_active,data_inert, eps,min_samples):
        points = [op.Point(*data_active[i,:].tolist()) for i in range(data_active.shape[0])]
        qq = op.Optics(points,eps,min_samples)
        qq.run()
        clusters = qq.cluster(eps)

        da = dict([(c,cluster) for c,cluster in enumerate(clusters)])
        di = dict([(c,cluster) for c,cluster in enumerate(clusters)])
        ret = {'data_active':da, 'data_inert':di}
        return ret
        
    # data_active is data to cluster, data_inert is data corresponding data_active but not to be used for clustering
    def dbscan(self,data_active,data_inert,eps,min_samples):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_active)
        clusterid = db.labels_
        clusters = list(set(clusterid))

        da = dict([(c,[]) for c in clusters])
        di = dict([(c,[]) for c in clusters])
        ret = {'data_active':da, 'data_inert':di}
        for c,d,dd in zip(clusterid,data_active,data_inert):
            ret['data_active'][c].append(d)
            ret['data_inert' ][c].append(dd)

        return ret

    def scatter(self,data,skip):
        # view pca
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        skip = 15
        ax.scatter(data[::skip,0],data[::skip,1],data[::skip,2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def makeClusterPolygons(self,clusters):
        # make polygons for each of those clusters
        clusterids = clusters['data_inert'].keys()
        if -1 in clusterids:
            clusterids.remove(-1)
        poly    =[]
        for k in clusterids:
            points  =np.asarray(clusters['data_inert'][k])
            if k>=0 and points.shape[0]>2:
                hull    =ConvexHull(points)
                hullvert    =list(hull.vertices)
                coords  =[(x,y) for x,y in points[hullvert,:]]
                poly.append((coords,))
        mp = MultiPolygon(poly)
        color = '#ff0000'
        category = 'pca'
        st = { "weight": 2, "color": color, "opacity": 0, "fillColor": color, "fillOpacity": 0.3 }
        pr = {"name":category, "popupContent":category,"style":st}
        ft = Feature(geometry=mp, properties=pr)

        outfilename = 'sample-geojson.js'
        outfile = open(outfilename,'w')
        print>>outfile, "var data = "
        geojson.dump(ft,outfile)
        #geojson.dump(ft,outfile)
        outfile.close()

    def scaleData(self,data,stdvs):
        ss = preprocessing.StandardScaler()
        datan = ss.fit_transform(data)
        datanc = np.clip(datan,-stdvs,stdvs)
        datancs = (datanc+stdvs)/(2*stdvs)

        return datancs
        
    def pcaanalysis(self,catfile,outfilename):
        # build a matrix with photos on rows, categories in columns
        categories = [line.split()[0] for line in open(catfile).readlines()]
        cat_matrix = []
        latlon = []
        for col in self.cols:
            docs = col.find({"$and":[{'prediction':{"$ne": 0}},{'prediction': {"$exists":True}}]},timeout=False)
            for i,doc in enumerate(docs):
                pca_row = [doc['prediction'][cat] for cat in categories]
                ll_row = [doc['longitude'],doc['latitude']]
                cat_matrix.append(pca_row)
                latlon.append(ll_row)
        
        matrix = np.log10(np.asarray(cat_matrix))
        mscaled = self.scaleData(matrix,3)
        npcoords = np.asarray(latlon)
        pca = self.pca(matrix,5)
        #ms = preprocessing.MinMaxScaler()
        #fc = ms.fit_transform(pca)
        fc = self.scaleData(pca,3)
        fc = np.power(10,fc-1.0)

        # pyplot
        plt.scatter(npcoords[:,0],npcoords[:,1],s=16,facecolors=fc[:,(0,1,2)],edgecolors='none')
        #plt.scatter(matrix[:,0],matrix[:,1],s=16,facecolors=mscaled[:,(2,3,4)],edgecolors='none')
        #plt.scatter(fc[:,2],fc[:,1],s=16,facecolors=fc[:,(0,3,4)],edgecolors='none')
        #plt.scatter(pca[:,2],pca[:,1],s=16,facecolors=fc[:,(0,3,4)],edgecolors='none')
        plt.show()

        # geojson
        g = gj.gjson()
        g.initFile('sample-geojson.js')
        latlon = np.asarray(latlon)
        g.writePointsFeatureCollection(latlon[:],fc[:,(0,1,2)],'data')
        g.closeFile()

if __name__=="__main__":
    db          ='geo'
    
    categoriesfile  ='../../caffe/models/placesCNN/categoryIndex_places205.csv'

    cols =[a.decode('utf-8') for a in sys.argv[1:]]
    print cols
    cutoff = 0.1
    eps = 0.005
    min_samples = 5
    outfilename = 'sample-geojson.js'
    
    a = plotstuff()
    a.setupDB(db,cols)
    a.getAllData(categoriesfile)
    #a.getCollectionStatistics(min_samples*3)
    b = a.getCollectionStatistics(min_samples)
    a.clusterCoordinates(b['pcntl'],min_samples)
    #eps = b['pcntl']
    #outfilename = 'sample-geojson_2.js'
    #a.makeFeatureCollection(categories,cutoff,eps,min_samples,outfilename)
    #a.makeMultiPointFeatureCollection(category,cutoff,eps,min_samples,outfilename)

    #a.pcaanalysis(categoriesfile,outfilename)
