# -*- coding: utf-8 -*-

import re
import json

#"train_data\\not_shared\\SARC2.0\\main\\comments.json"
def readbyparts_yielded(filename="train_data\\not_shared\\Folder_NLPEnglish_Dialogs\\SARC2.0\\pol\\comments.json", step=500):
    regex_str = '\"[a-zA-Z0-9]+\":\ \{[^\{]+\}'
    fileobj = open(filename, "r")
    #readed
    read_string = fileobj.read(step)
    while read_string != '':
        match_obj = re.search(regex_str, read_string)
        while match_obj != None:
            obj_str = "{"+read_string[match_obj.start():match_obj.end()]+"}"
            obj = json.loads(obj_str)
            _id = list(obj.keys())[0]
            obj = obj[_id]
            obj["_id"] = _id
            #print(o)
            yield obj
            #delete already parsed part
            read_string = read_string[match_obj.end():]
            match_obj = re.search(regex_str, read_string)
        read_string = read_string + fileobj.read(step)
    fileobj.close()

#example first 2 objects:
#{
#    'text': 'Nancyt Pelosi messes up.. 500 Million Jobs lost every month that the economic recovery plan is not passed.. LMAO',
#    'author': 'Fishbum',
#    'score': 0,
#    'ups': 2,
#    'downs': 4,
#    'date': '2009-02',
#    'created_utc': 1233788424,
#    'subreddit': 'politics',
#    '_id': '7uxqr'
#},
#{
#    'text': 'Netflix CEO: "Please raise my taxes"',
#    'author': 'jdl2003',
#    'score': 1733,
#    'ups': 1985,
#    'downs': 252,
#    'date': '2009-02',
#    'created_utc': 1233940024,
#    'subreddit': 'politics',
#    '_id': '7vewt'
#}