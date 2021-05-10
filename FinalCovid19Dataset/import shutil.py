import shutil
import os
source = os.listdir("COVID")
destination = "test/0"
path = os.getcwd()
i = 0
for files in source:
    if files.endswith(".png") and i < 251:
        shutil.move("{}/COVID/{}".format(path,files),"{}/{}".format(path,destination))
        i = i + 1