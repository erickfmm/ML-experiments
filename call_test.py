from os import listdir
from os.path import join
import subprocess
import sys

test_files = listdir("test")

i = 0
for test_file in test_files:
    if test_file.endswith(".py"):
        print(i, " - ", test_file)
    i += 1

which_idx = int(input("Id of file: "))

#subprocess.run(["pip", "list", "--format=freeze", ">", "data/sample_data/reqs.txt" ])
subprocess.run([sys.executable, join("test", test_files[which_idx])])
