import time
from glob import glob
from random import shuffle
from contextlib import contextmanager
import json
import base64

@contextmanager
def timer(name):

    t0 = time.time()
    yield
    print("[{}] done in {} sec.".format(name, round(time.time() - t0)))



def getimages():
    imgs = glob("./dataset/*.jpeg")
    shuffle(imgs)
    for img in imgs:
        print("image file", img)
        with open(img, 'rb') as image:
            image_read = image.read()
        yield base64.b64encode(image_read).decode('utf-8') #encodestring also works aswell as decodestring
        



        
