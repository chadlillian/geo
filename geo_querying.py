#!/home/chad/anaconda/bin/python
import  sys
from    pymongo import MongoClient
import  numpy as np

class   querying:
    def __init__(self):
        self.client =MongoClient()

    def getCollections(self,db):
        self.db     =self.client[db]

    def setupDB(self,db,cols):
        self.db     =self.client[db]
        self.cols = [self.db[col] for col in cols]

    # get all coordinates, predictions, and photo urls from the pre-selected collection
    def getAllData(self,catfile):
        categories = [line.split()[0] for line in open(catfile).readlines()]
        data = []
        coords = []
        photos = []
        for col in self.cols:
            #docs = col.find({"$and":[{'prediction':{"$ne": 0}},{'prediction': {"$exists":True}}]},timeout=False)
            docs = col.find({"$and":[{'prediction':{"$ne": 0}},{'prediction': {"$exists":True}}]})

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

        return {'predictions':npdata,'coordinates':npcoords,'photos':photos}

if __name__=="__main__":
    db          ='geo'
    
    categoriesfile  ='../../caffe/models/placesCNN/categoryIndex_places205.csv'

    cols =[a.decode('utf-8') for a in sys.argv[1:]]
    print cols
    a = querying()
    a.setupDB(db,cols)
    a.getAllData(categoriesfile)
