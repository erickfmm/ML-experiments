# -*- coding: utf-8 -*-

import load_data.loader.text._ENG_reddit_comments as rcs

l = rcs.LoadRedditComments(onlyfiles=[])

print("load into mongo")
# l.load_comments_into_mongo(insert_from=7610000)
l.load_comments_into_mongo()
print("make pairs")
l.make_pairs_into_mongo()
print("write pairs")
l.write_pairs_into_files()
print("done")