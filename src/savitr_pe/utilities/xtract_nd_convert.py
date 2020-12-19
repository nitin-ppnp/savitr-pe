#!/usr/bin/python
import sys
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import sys
import struct
import cv2
import time
import math
import glob
from multiprocessing import Pool
from zipfile import ZipFile


_display = False
def myencode(imgname):
    st = time.time()

    f = open(imgname, "rb")

    # java: 32 bit ints -> 4 bytes each
    w = int(struct.unpack('>i', f.read(4))[0])
    h = int(struct.unpack('>i', f.read(4))[0])
    pstride = int(struct.unpack('>i', f.read(4))[0])
    rstride = int(struct.unpack('>i', f.read(4))[0])

    if _display:
        print ("width: ", w )
        print ("height: ",h) 
        print ("pstride: ",pstride )
        print ("rstride: ", rstride) 

    a = np.fromfile(f, dtype=np.uint8)
    f.close()

    Y = a[0:(w*h)].reshape(h,w)

    if _display:
        fig = plt.figure(0)
        plt.imshow(Y, cmap='gray')

        print ("Remaining: ", a[(w*h):].shape)

    U = np.zeros((h,w), np.uint8)
    r=0
    l=w
    
    for i,p in enumerate(a[(w*h)-1:int((w*h)-1+(w/pstride*h)+1)]):
        U[r, int((i-r*l)/pstride) ] = p
        r = int(i/l)
        
    if _display:
        plt.figure(1)
        plt.imshow(U, vmin=0,vmax=255,interpolation='none')
        plt.title("U")


    V = np.zeros((h,w), np.uint8)
    r=0
    l=w
    for i,p in enumerate(a[int((w*h)-1+(w/pstride*h)-1):]):
        V[r,int((i-r*l)/pstride)] = p
        r = int(i/l)

    if _display:
        plt.figure(2)
        plt.imshow(V, vmin=0,vmax=255,interpolation='none')
        plt.title("V")
        plt.show()

    #print "Until here: ", (time.time() - st)

    # resample images to vectorize combination
    import scipy.ndimage
    U_ = scipy.ndimage.zoom(U[:int(h/pstride),:int(w/pstride)],2,order=0)
    V_ = scipy.ndimage.zoom(V[:int(h/pstride),:int(w/pstride)],2,order=0)

    Y = Y - 16.
    V_ = V_ - 128.
    U_ = U_ - 128.

    rgb = np.zeros((h,w,3), np.uint8)

    rgb[:,:,0] = np.clip(1.164*Y + 1.596*V_, 0, 255)
    rgb[:,:,1] = np.clip(1.164*Y - 0.813*V_ - 0.391*U_, 0, 255)
    rgb[:,:,2] = np.clip(1.164*Y + 2.018*U_, 0, 255)


    #cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    #cv2.waitKey(10)

    outname = imgname[:-4] + ".png"
    # print ("outname: ", outname, " took ", (time.time() - st))
    cv2.imwrite(outname, cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))

    if _display:
        plt.figure(3)
        plt.title("RGB")
        plt.imshow(rgb)
        plt.show()


# extract files
zipfilenames = glob.glob(sys.argv[1] + "/" + "*.zip")

for fname in zipfilenames:
    with ZipFile(fname,"r") as f:
        f.extractall(fname[:-4])

print("############### File extraction complete ###########################")
print("###################################################################")


imdirs = [os.path.join(sys.argv[1],f) for f in os.listdir(sys.argv[1]) if os.path.isdir(os.path.join(sys.argv[1],f))]

for imdir in imdirs:

    filenames = glob.glob(imdir + "/" + "*.yuv")
    print ("Got %d files, decoding now" % len(filenames))

    stt = time.time()
    for filename in filenames:
        myencode(filename)
    
    print ("Decoding ,",len(filenames)," images took ", time.time() - stt)
    
