#!/home/chad/anaconda/bin/python
import numpy as np
import sys
from   sklearn.cluster import DBSCAN
from   sklearn.manifold import TSNE
import optics as op
import matplotlib.pyplot as plt
import matplotlib.figure as pltfig
from   mpl_toolkits.mplot3d   import Axes3D
from	sklearn.decomposition	import	PCA
from    sklearn import preprocessing
import alphashape
import geo_json as gj

class   clustering:
    def __init__(self):
        return

    def optics(self,data_active,data_inert, eps,min_samples):
        points = [op.Point(*data_active[i,:].tolist()) for i in range(data_active.shape[0])]
        qq = op.Optics(points,eps,min_samples)
        qq.run()
        clusters = qq.cluster(eps)

        da = dict([(c,cluster) for c,cluster in enumerate(clusters)])
        di = dict([(c,cluster) for c,cluster in enumerate(clusters)])
        ret = {'data_active':da, 'data_inert':di}
        self.ret = ret
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

        for c in clusterid:
            ret['data_active'][c] = np.copy(np.array(ret['data_active'][c]))
            ret['data_inert'][c] = np.copy(np.array(ret['data_inert'][c]))

        self.ret = ret
        return ret

    def plot(self):
        da = self.ret['data_active']
        di = self.ret['data_inert']
        p = dict([ (x,di[x].max(1)) for x in di.keys() if x>=0])
        q = [ di[x].max(0) for x in di.keys() if x>=0]
        r = np.asarray(q)
        pca = self.pca(r,7)
        qqq = self.arrayToRGB(pca[:,0:3])
        model = TSNE(n_components=2,random_state=0)
        tsne = model.fit_transform(pca)
        plt.scatter(tsne[:,0],tsne[:,1],c=qqq,edgecolors='none')
        plt.show()

        q = [ [len(da[x]),x,np.mean(da[x],0),np.linalg.norm(np.std(da[x],0)) ] for x in da.keys() if x>=0]
        q.sort(reverse=True)

        x = [qq[2][0] for qq in q]
        y = [qq[2][1] for qq in q]
        s = [(2100**2)*3.1415*qq[3]**2 for qq in q]

        #fig = plt.figure()
        #ax = fig.add_subplot(111,projection='3d')
        #ax.scatter(pca[:,0],pca[:,1],pca[:,2],c=qqq,edgecolors='none')
        plt.scatter(pca[:,0],pca[:,1],c=qqq,edgecolors='none')
        plt.show()
        plt.scatter(y,x,s,color = qqq, edgecolors='none')
        plt.show()

        return zip(x,y,s)
        
    def pca(self,data,dim):
        pca_matrix = np.log10(np.asarray(data))
        pca_ = PCA(n_components=dim)
        pca_.fit(pca_matrix)
        b = pca_.transform(pca_matrix)

        return b

    def arrayToRGB(self,a):
        ms = preprocessing.MinMaxScaler()
        fc = ms.fit_transform(a)
        hxcolor = ["#%02x%02x%02x"%tuple(map(int,255*f)) for f in fc]
        #hxcolor = ["#%02x%02x%02x"%tuple(map(int,255*np.array([f[0],0,0]))) for f in fc]
        #hxcolor = ["#%02x%02x%02x"%tuple(map(int,255*np.array([0,f[1],0]))) for f in fc]
        return hxcolor

    def makeJson(self):
        da = self.ret['data_active']
        di = self.ret['data_inert']
        q = [ di[x].max(0) for x in di.keys() if x>=0]
        r = np.asarray(q)
        qq = self.pca(r,3)
        qqq = self.arrayToRGB(qq)

        polygons = [ [alphashape.alpha_shape_wrapper(da[x][:,(1,0)],50.0)] for x in da.keys() if x>=0]
        opacity = [0]*len(polygons)

        print polygons[0]
        gjson = gj.gjson()
        gjson.initFile("data.json")
        gjson.writeMultiPolygonFeatureCollection_(polygons,fillColor=qqq,color=qqq)#,opacity=opacity)
        gjson.closeFile()
