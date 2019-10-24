import pymongo
#

class VectorizedWordsTxtMongoDB:
    def __init__(self, dbname, server_ip="localhost", server_port=27017): #dbname="wikivec"
        self.mongo_client = pymongo.MongoClient("mongodb://"+server_ip+":"+str(server_port))
        self.db = self.mongo_client[dbname]
        #mycol = mydb["customers"]
    
    def closeConnection(self):
        self.mongo_client.close()
    
    #loadIntoMongoDB("train_data\\not_shared\\wikivec\\wiki-news-300d-1M.vec", "withoutsubwords")
    #loadIntoMongoDB("train_data\\not_shared\\wikivec\\wiki-news-300d-1M-subword.vec", "withsubwords")
    def loadIntoMongoDB(self, vecFile, collectionName, verbose_mod=10000, n_insert=100):
        collist = self.db.list_collection_names()
        n_to_insert=0
        to_insert=[]
        if collectionName in collist:
            print("The collection exists.")
            return -1
        collection = self.db[collectionName]
        with open(vecFile, "r", encoding='utf-8') as filevec:
            iline = 0
            for line in filevec:
                if verbose_mod is not None and iline % verbose_mod == 0:
                    print("already inserted: ", iline)
                if iline > 0:
                    array_data = line.strip().split(" ")
                    name = array_data[0]
                    array_data = [float(array_data[i]) for i in range(1, len(array_data))]
                    #collection.insert_one({"_id": name, "value": array_data})
                    if n_to_insert == n_insert:
                        collection.insert_many(to_insert)
                        to_insert = []
                        n_to_insert = 0
                    else:
                        to_insert.append({"_id": name, "value": array_data})
                        n_to_insert +=1
                iline += 1
            if n_to_insert > 0:
                collection.insert_many(to_insert) #remain
                to_insert = []
                n_to_insert = 0
        print("done")
        return 0
    
    def GetWord(self, collectionName, word):
        collection = self.db[collectionName]
        word_docs = collection.find({"_id": word})
        array_docs = []
        for doc in word_docs:
            array_docs.append(doc)
        if len(array_docs) == 1:
            return array_docs[0]["value"]
        elif len(array_docs) > 1:
            return len(array_docs)
        else:
            return None