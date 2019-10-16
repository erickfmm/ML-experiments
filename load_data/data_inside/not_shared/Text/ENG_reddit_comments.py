# -*- coding: utf-8 -*-


import json

def is_acceptable(obj):
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
    elif len(obj["body"].split(" ")) > 50: #because one model
        return False
    elif obj["edited"] == True:
        return False
    elif obj["score"] < 2: #for relevant comments
        return False
    else:
        return True

def read_file(filepath="train_data\\not_shared\\Folder_NLPEnglish_Dialogs\\Reddit comments\\RC_2005-12"):
    with open(filepath, "r") as fobj:
        nline = 0
        with_parent = 0
        for line in fobj:
            nline +=1
            #yield json.loads(line.strip())
            json_obj = json.loads(line.strip())
            json_obj["body"] = format_comment(json_obj["body"])
            if is_acceptable(json_obj):
                #print("reading: ", nline)
                json_obj = {"_id": json_obj["id"],
                          "parent_id": json_obj["parent_id"],
                          "body": json_obj["body"],
                          "subreddit": json_obj["subreddit"],
                          "score": json_obj["score"],
                          "created_utc": json_obj["created_utc"]
                          #,"controversiality": json_obj["controversiality"]
                            }
                pid = json_obj["parent_id"].split("_")
                if pid[0] == "t1":# or pid[0] == "t2":
                    json_obj["parent_id"] = pid[1]
                    with_parent +=1
                else:
                    json_obj.pop("parent_id")
                if len(json_obj["body"])>0: #check again because of format_comment changes
                    yield json_obj
        print("total lines: ", nline)
        print("with parent: ", with_parent)


import re

def del_markdown_links(comment, intermediate_char=" "):
    #get rid markdown links
    # Anything that isn't a square closing bracket
    name_regex = "[^]]+"
    # http:// or https:// followed by anything but a closing paren
    url_regex = "http[s]?://[^)]+"
    
    markup_regex = '\[({0})]\(\s*({1})\s*\)'.format(name_regex, url_regex)
    for match in re.findall(markup_regex, comment):
        comment = comment.replace(match[1], "")
        while True:
            intermed_idx = comment.find("]()")
            #stops when no more "]()", because may be more than 1 
            #links equal and replace deletes them all
            if intermed_idx == -1:
                break
            #inverse the array to select the "[" just before the "]()",
            #so its the beggining of link structure [desc](link)
            begin_indx = comment[0:intermed_idx][::-1].find("[")
            if begin_indx != -1:
                #the way to delete charatcers by index is slicing:
                #s = s[:idx]+s[idx+1:]
                #
                #the begin_indx was taken in inversed string, 
                #so needs to be adjusted using such formula.
                comment = comment[:intermed_idx-begin_indx-1]+comment[intermed_idx-begin_indx:]
                intermed_idx = comment.find("]()")
                comment = comment[:intermed_idx]+intermediate_char+comment[intermed_idx+3:]
    return comment

def format_comment(comment):
    comment = del_markdown_links(comment)
    #tokenize newchar and replace " with ' to normalize 
    #(may be make a problem with 's and 've, etc)
    return comment.replace("\n", " newlinechar ").replace("\r", " newlinechar ").replace('"', "'").strip()

    
    
#if __name__ == "__main__":
datas = dict()
for obj in read_file():
    datas[obj["_id"]] = obj
print("total fetched: ", len(datas))
i = 0
pairs = []
for key in datas:
    d = datas[key]
    if "parent_id" in d:
        if d["parent_id"] in datas:
            #print("######", key)
            #print(datas[d["parent_id"]]["body"], "\n -> \n", d["body"])
            pairs.append((datas[d["parent_id"]], d))
        i+=1
    
print("pairs: ", len(pairs))
#sample coment
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