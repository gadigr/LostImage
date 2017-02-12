
from PIL import Image
import os, re
from os import listdir
from os.path import basename

pat = "^(\d|\d\d)\.jpg$"

def purge(dir, pattern):
    for f in os.listdir(dir):
        if not re.search(pattern, f):
            print(os.path.join(dir, f))
            os.remove(os.path.join(dir, f))

dirs = listdir('letters/')
for directory in dirs:
    purge('letters/'+directory, pat)
    # images = listdir('letters/'+directory)
    # i = 1
    # for img in images:
    #     fd_img = 'letters/'+directory + '/' + img
    #     # imgg = Image.open(fd_img)
    #     # # print(os.path.splitext(fd_img)[0] + '.jpg')
    #     # imgg.save('letters/'+directory + '/' + str(i) + '.jpg')
    #     # i = i + 1
        
    #     if (re.search(pattern, fd_img)):
    #         print (fd_img)

print('done')