import re
import pymongo
import json
from os.path import join, exists
from os import listdir, makedirs
import datetime

__all__ = ["LoadRedditComments"]


def del_markdown_links(comment, id_comment=None, intermediate_char=" "):
    # get rid markdown links
    # Anything that isn't a square closing bracket
    name_regex = "[^]]+"
    # http:// or https:// followed by anything but a closing paren
    url_regex = "http[s]?://[^)]+"
    
    markup_regex = '\[({0})]\(\s*({1})\s*\)'.format(name_regex, url_regex)
    deleted_matches = 0
    iterations_run = 0
    for match in re.findall(markup_regex, comment):
        comment = comment.replace(match[1], "")
        while True:
            intermed_idx = comment.find("]()")
            # stops when no more "]()", because may be more than 1
            # links equal and replace deletes them all
            if intermed_idx == -1:
                break
            # inverse the array to select the "[" just before the "]()",
            # so its the beggining of link structure [desc](link)
            begin_indx = comment[0:intermed_idx][::-1].find("[")
            # if intermed_idx != -1 and begin_indx == -1:
            # in already processed:  7130000  and inserted 7129938
            # file 2015-1, some file has "]()" so not entered previous break
            # prev file is around 100.000 already processed aprox
            # but doesn't have begin_indx so doesn't enter if down
            # so it loops endlessly
            #    break
            if begin_indx != -1:
                # the way to delete charatcers by index is slicing:
                # s = s[:idx]+s[idx+1:]
                #
                # the begin_indx was taken in inversed string,
                # so needs to be adjusted using such formula.
                comment = comment[:intermed_idx-begin_indx-1]+comment[intermed_idx-begin_indx:]
                intermed_idx = comment.find("]()")
                comment = comment[:intermed_idx]+intermediate_char+comment[intermed_idx+3:]
                deleted_matches += 1
            iterations_run +=1
            if deleted_matches != iterations_run:
                print("something happened in id: ", id_comment)
                print("comment: ", comment)
                return None
    return comment


class LoadRedditComments:
    def __init__(self, dbname="redditcomments", server_ip="localhost", server_port=27017
                 , min_score=2
                 , folder_path="train_data\\Folder_NLPEnglish_Dialogs\\Reddit comments\\data"
                 , onlyfiles=None
                 , logfile="train_data\\Folder_NLPEnglish_Dialogs\\Reddit comments\\logfile.log"):
        if onlyfiles is None:
            onlyfiles = ["RC_2005-12"]
        self.mongo_client = pymongo.MongoClient("mongodb://"+server_ip+":"+str(server_port))
        self.db = self.mongo_client[dbname]
        self.min_score = min_score
        self.folder_path=folder_path
        self.onlyfiles = onlyfiles if len(onlyfiles)>0 else listdir(folder_path)
        self.logobj = open(logfile, "a")
        self.logobj.write("\n\n\n\n")
        self.logobj.write("#"*40+"\n")
        self.logobj.write("#"*40+"\n")
        self.logobj.write(str(datetime.datetime.now())+"\n")
    
    def drop_database(self, dbname="redditcomments"):
        self.mongo_client.drop_database(dbname)
    
    def close_connection(self):
        try:
            self.mongo_client.close()
        except:
            print("closing mongodb")
        try:
            self.logobj.flush()
            self.logobj.close()
        except:
            print("error closing logfile")
        try:
            # flush all files
            self.from_train_file.flush()
            self.to_train_file.flush()
            self.from_train_ids_file.flush()
            self.to_train_ids_file.flush()
            # close all files
            self.from_train_file.close()
            self.to_train_file.close()
            self.from_train_ids_file.close()
            self.to_train_ids_file.close()
        except Exception as e:
            print("error closing write files", e)
    
    def load_comments_into_mongo(self, verbose_mod=10000, n_insert=100, insert_from=0, skip_first_lines=0):
        self.logobj.write("\n\n\n\n")
        self.logobj.write("#"*40+"\n")
        self.logobj.write("#"*40+"\n")
        self.logobj.write("to load comments into mongo.\n")
        collection = self.db["comments"]
        n_to_insert = 0
        to_insert = []
        iline = 0
        total_inserted = 0
        for filename in self.onlyfiles:
            fullname = join(self.folder_path, filename)
            if filename[0:3] == "RC_" and exists(fullname):
                print("file: ", filename)
                self.logobj.write("file: " + filename)
                for obj in self.read_file(fullname, skip_first_lines):
                    iline +=1
                    # collection.insert_one(obj)
                    if verbose_mod is not None and iline % verbose_mod == 0:
                        print("already processed: ", iline, " and inserted", total_inserted)
                        self.logobj.write("already processed: " + str(iline) +
                                          " and inserted" + str(total_inserted) + "\n")
                    if iline < insert_from:
                        continue
                    if n_to_insert == n_insert:
                        try:
                            collection.insert_many(to_insert)
                        except Exception as e:
                            print("error inserting in line ", iline, " error: ", e)
                            self.logobj.write("error inserting in line " + str(iline)+"\n")
                        total_inserted += n_to_insert
                        to_insert = []
                        n_to_insert = 0
                    to_insert.append(obj)
                    n_to_insert += 1
                if n_to_insert > 0:
                    try:
                        collection.insert_many(to_insert)
                    except Exception as e:
                        print("error inserting in line ", iline, " error: ", e)
                        self.logobj.write("error inserting in line (when end file) " + str(iline)+"\n")
                    total_inserted += n_to_insert
                    to_insert = []
                    n_to_insert = 0
                    print("end file")
                    print("already processed: ", iline, " and inserted ", total_inserted)
                    self.logobj.write("end file\n processed " + str(iline)+" and inserted " + str(total_inserted)+"\n")
        if verbose_mod is not None:
            print("read and insert done")
            print("total processed: ", iline, " and inserted", total_inserted)
            self.logobj.write("read and insert done\n")
            self.logobj.write("total processed: " + str(iline) + " and inserted"+str(total_inserted) + "\n")

    def make_pairs_into_mongo(self, verbose_mod=10000, skip_first_comments=0):
        self.logobj.write("\n\n\n\n")
        self.logobj.write("#"*40+"\n")
        self.logobj.write("#"*40+"\n")
        self.logobj.write("to make pairs into mongo.\n")
        print("to make pairs into mongo.")
        collection_comments = self.db["comments"]
        collection_pairs_meta = self.db["pairs_meta"]
        icomment = 0
        for comment in collection_comments.find():
            icomment += 1
            if icomment < skip_first_comments:
                if icomment % verbose_mod == 0:
                    print(str(icomment)+" skipped when make pairs into mongo.")
                    self.logobj.write(str(icomment)+" skipped when make pairs into mongo.\n")
                continue
            if icomment % verbose_mod == 0:
                print(str(icomment)+" pairs into mongo.")
                self.logobj.write(str(icomment)+" pairs into mongo.\n")
            if "parent_id" in comment:
                parent_comment = collection_comments.find_one({"_id": comment["parent_id"]})
                if parent_comment is not None:
                    parent_responses = collection_pairs_meta.find_one({"_id": parent_comment["_id"]})
                    if parent_responses is not None:
                        # test if its already in
                        if comment["_id"] in parent_responses["responses"]:
                            continue
                        all_responses = parent_responses["responses"].copy()
                        all_responses.append(comment["_id"])
                        best_id = parent_responses["best_response"]
                        best_score = parent_responses["best_score"]
                        if comment["score"] > parent_responses["best_score"]:
                            best_id = comment["_id"]
                            best_score = comment["score"]
                        collection_pairs_meta.update_one({"_id": parent_comment["_id"]},
                                                          {"$set": {
                                                                  "responses": all_responses,
                                                                  "best_score": best_score,
                                                                  "best_response": best_id}})
                    else: # no responses founded
                        collection_pairs_meta.insert_one({"_id": parent_comment["_id"],
                                                          "responses": [comment["_id"]],
                                                          "best_score": comment["score"],
                                                          "best_response": comment["_id"]})
        print("pairs done")
        self.logobj.write("pairs done\n")

    def write_pairs_into_mongo_col(self, verbose_mod=10000):
        self.logobj.write("\n\n\n\n")
        self.logobj.write("#"*40+"\n")
        self.logobj.write("#"*40+"\n")
        self.logobj.write("to write pairs into mongodb full comments using pairs collection\n")
        collection_pairs_full = self.db["full_pairs"]
        iwrite=0
        collection_pairs_meta = self.db["pairs_meta"]
        collection_comments = self.db["comments"]
        for pair in collection_pairs_meta.find():
            parent_comment = collection_comments.find_one({"_id": pair["_id"]})
            comment = collection_comments.find_one({"_id": pair["best_response"]})
            collection_pairs_full.insert_one({
                    "_id": pair["_id"],
                    "from": parent_comment["body"],
                    "to": comment["body"]
                    })
            if iwrite % verbose_mod == 0:
                print("writen: ", iwrite)
                self.logobj.write("written in mongo: "+str(iwrite)+"\n")
                self.logobj.flush()
            iwrite += 1

    def write_pairs_into_files(self,
                               folder_pairs="train_data\\Folder_NLPEnglish_Dialogs\\Reddit comments\\result",
                               n_write=100,
                               verbose_mod=10000):
        self.logobj.write("\n\n\n\n")
        self.logobj.write("#"*40+"\n")
        self.logobj.write("#"*40+"\n")
        self.logobj.write("to write pairs files\n")
        if not exists(folder_pairs):
            makedirs(folder_pairs)
        self.from_train_file = open(join(folder_pairs, "train.from"), "w")
        from_train_lines = []
        self.to_train_file = open(join(folder_pairs, "train.to"), "w")
        to_train_lines = []
        self.from_train_ids_file = open(join(folder_pairs, "train_ids.from"), "w")
        from_train_ids_lines = []
        self.to_train_ids_file = open(join(folder_pairs, "train_ids.to"), "w")
        to_train_ids_lines = []
        # from_test_file = open(join(folder_pairs, "test.from"), "w")
        # to_test_file = open(join(folder_pairs, "test.to"), "w")
        iwrite=0
        collection_pairs_meta = self.db["pairs_meta"]
        collection_comments = self.db["comments"]
        for pair in collection_pairs_meta.find():
            parent_comment = collection_comments.find_one({"_id": pair["_id"]})
            comment = collection_comments.find_one({"_id": pair["best_response"]})
            # append lines
            # from_train_lines.append(parent_comment["body"]+"\n")
            # to_train_lines.append(comment["body"]+"\n")
            # from_train_ids_lines.append(parent_comment["_id"]+"\n")
            # to_train_ids_lines.append(comment["_id"]+"\n")
            # write directly
            try:
                self.from_train_ids_file.write(parent_comment["_id"]+"\n")
                self.to_train_ids_file.write(comment["_id"]+"\n")
                self.from_train_file.write(parent_comment["body"]+"\n")
                self.to_train_file.write(comment["body"]+"\n")
            except:
                import sys
                print("error")
                print(parent_comment)
                print("->")
                print(comment)
                raise sys.last_value
            iwrite += 1
            if True:  # TODO: False  # iwrite % n_write == 0:
                print(iwrite)
                print([len(from_train_lines), len(to_train_lines), len(from_train_ids_lines), len(to_train_ids_lines)])
                self.from_train_ids_file.writelines(from_train_ids_lines)
                self.to_train_ids_file.writelines(to_train_ids_lines)
                self.from_train_file.writelines(from_train_lines)
                self.to_train_file.writelines(to_train_lines)
                # reset all
                from_train_ids_lines = []
                to_train_ids_lines = []
                from_train_lines = []
                to_train_lines = []
                # flush all
                self.from_train_ids_file.flush()
                self.to_train_ids_file.flush()
                self.from_train_file.flush()
                self.to_train_file.flush()
            if iwrite % verbose_mod == 0:
                print("written in files: "+str(iwrite)+" lines")
                self.logobj.write("written in files: "+str(iwrite)+" lines\n")
                self.logobj.flush()
        # write remain lines
        # self.from_train_file.writelines(from_train_lines)
        # self.to_train_file.writelines(to_train_lines)
        # self.from_train_ids_file.writelines(from_train_ids_lines)
        # self.to_train_ids_file.writelines(to_train_ids_lines)
        # flush all files
        self.from_train_file.flush()
        self.to_train_file.flush()
        self.from_train_ids_file.flush()
        self.to_train_ids_file.flush()
        # close all files
        self.from_train_file.close()
        self.to_train_file.close()
        self.from_train_ids_file.close()
        self.to_train_ids_file.close()
        print("write done")
        self.logobj.write("write done\n")
        self.logobj.flush()

    def is_acceptable(self, obj):
        if "body" not in obj:
            return False
        elif obj["body"] == "[deleted]":
            return False    
        elif obj["body"] == "[removed]":
            return False
        elif len(obj["body"]) < 1:
            return False
        elif len(obj["body"]) > 1000:
            return False
        elif len(obj["body"].split(" ")) > 50:  # because one model
            return False
        elif obj["edited"]:  # if True
            return False
        elif obj["score"] < self.min_score:  # for relevant comments
            return False
        else:
            return True
    
    def read_file(self, filepath, skip_first_lines=0):
        with open(filepath, "r") as fobj:
            self.logobj.write("open "+filepath)
            nline = 0
            with_parent = 0
            for line in fobj:
                nline +=1
                if nline < skip_first_lines:
                    continue
                # yield json.loads(line.strip())
                json_obj = json.loads(line.strip())
                json_obj["body"] = self.format_comment(json_obj["body"],
                        [nline, " id: ", json_obj["id"],  " of ", filepath, " utc ", json_obj["created_utc"] ])
                if json_obj["body"] is None:
                    self.logobj.write("error in format comment\n")
                    self.logobj.write(str(nline) + " id: " + str(json_obj["id"]) + " of "
                                      + filepath + " utc " + str(json_obj["created_utc"])+"\n")
                    continue
                if self.is_acceptable(json_obj):
                    # print("reading: ", nline)
                    json_obj = {"_id": json_obj["id"],
                                "parent_id": json_obj["parent_id"],
                                "body": json_obj["body"],
                                "subreddit": json_obj["subreddit"],
                                "score": json_obj["score"],
                                "created_utc": json_obj["created_utc"]
                                # ,"controversiality": json_obj["controversiality"]
                                }
                    pid = json_obj["parent_id"].split("_")
                    if pid[0] == "t1":  # or pid[0] == "t2":
                        json_obj["parent_id"] = pid[1]
                        with_parent +=1
                    else:
                        json_obj.pop("parent_id")
                    if len(json_obj["body"]) > 0:  # check again because of format_comment changes
                        yield json_obj
            print("total lines: ", nline)
            print("with parent: ", with_parent)
            self.logobj.write("total lines: "+ str(nline)+"\n")
            self.logobj.write("with parent: "+ str(with_parent)+"\n")
    

    @staticmethod
    def format_comment(comment, id_comment=None):
        comment = del_markdown_links(comment, id_comment)
        if comment is None:
            return None
        # tokenize newchar and replace " with ' to normalize
        # (may be make a problem with 's and 've, etc)
        return comment.replace("\n", " newlinechar ").replace("\r", " newlinechar ").replace('"', "'").strip()


if __name__ == "__main__":
    # if False:
    datas = dict()
    load_reddit_obj = LoadRedditComments()
    for obj in load_reddit_obj.read_file("train_data\\Folder_NLPEnglish_Dialogs\\Reddit comments\\data\\RC_2005-12"):
        datas[obj["_id"]] = obj
    print("total fetched: ", len(datas))
    i = 0
    pairs = []
    for key in datas:
        d = datas[key]
        if "parent_id" in d:
            if d["parent_id"] in datas:
                # print("######", key)
                # print(datas[d["parent_id"]]["body"], "\n -> \n", d["body"])
                pairs.append((datas[d["parent_id"]], d))
            i += 1
        
    print("pairs: ", len(pairs))
# sample comment
"""
{
'archived': False,
'author': 'YoungModern',
'author_flair_css_class': None,
'author_flair_text': None,
'body': 'Most of us have some family members like this. *Most* of my '
        'family is like this. ',
'controversiality': 0,
'created_utc': '1420070400',
'distinguished': None,
'downs': 0,
'edited': False,
'gilded': 0,
'id': 'cnas8zv',
'link_id': 't3_2qyr1a',
'name': 't1_cnas8zv',
'parent_id': 't3_2qyr1a',
'retrieved_on': 1425124282,
'score': 14,
'score_hidden': False,
'subreddit': 'exmormon',
'subreddit_id': 't5_2r0gj',
'ups': 14}

"""