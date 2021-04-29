import pymongo


class Db:
    def __init__(self):
        # client = pymongo.MongoClient('localhost', 27017)
        client = pymongo.MongoClient("mongodb+srv://andreas:paokpaok@tacluster.litkf.mongodb.net/tripadvisor?retryWrites=true&w=majority")
        self.db = client['tripadvisor']

    def find_all(self, collection_name: str):
        return list(self.db[collection_name].find({}, {'_id': False}))

    def find(self, collection_name:str, query:dict):
        return list(self.db[collection_name].find(query, {'_id': False}))