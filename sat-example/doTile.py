#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""


from utils.timer import Timer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.io as sio
import argparse
from contextlib import contextmanager
import shutil
from subprocess import Popen, PIPE
import shlex
import tempfile
import re
import time
import fcntl
import os
import sys
from timeit import default_timer as timer

from osgeo import gdal


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--tiles', dest='tile_path',
                        help='image tile output path',
                        default=None, type=str)

    parser.add_argument('dstring', help="Detection string to  process",
                        type=str)
    args = parser.parse_args()

    return args

def mergeTiles(src, dst):

    img1 = mpimg.imread(src)
    img2 = mpimg.imread(dst)

    # Alpha Blend the two tiles
    src_rgb = img1[..., :3]
    src_a = img1[...,3]
    dst_rgb = img2[..., :3]
    dst_a = img2[...,3]
    out_a = src_a + dst_a*(1.0-src_a)
    out_rgb = (src_rgb*src_a[..., None] +
               dst_rgb*dst_a[..., None] * (1.0-src_a[..., None]))/ out_a[..., None]
    out = np.zeros_like(img1)
    out[..., :3] = out_rgb
    out[...,3] = out_a
    mpimg.imsave(dst, out)

def moveTiles(src, dst):
    files = os.listdir(src)
    if not os.path.exists(dst):
        os.makedirs(dst)
        # chmod of dirs
        for p,d,f in os.walk(dst):
            os.chmod(p, 0o777)
    for f in files:
        sname = os.path.join(src, f)
        dname = os.path.join(dst, f)
        if os.path.isdir(sname):
            moveTiles(sname, dname)
        else:
            fname, ext = os.path.splitext(dname)
            # Currently only moving the tiles since the
            # tilemapresource.xml is not being used by leaflet.
            # TODO: merge the tilemapresource.xml files by
            # reading the xml and updating the bounding box, and
            # x,y of the tiles.
            if os.path.exists(dname) == True and ext == '.png':
                mergeTiles(sname, dname)

                #i = 0;
                #dname2 = dname + str(i)
                #while os.path.exists(dname2) == True:
                #    i += 1
                #    dname2 = dname + str(i)
                #shutil.move(sname, dname2)

                #os.chmod(dname, 0o666)
                pass
            elif ext == '.png':
                shutil.move(sname, dname)
                os.chmod(dname, 0o666)

def parseRectStr(rectStr):
    items = rectStr.split(' ')
    # the first item is the class which we will ignore
    x = int(round(float(items[1])))
    y = int(round(float(items[2])))
    w = int(round(float(items[3])))
    h = int(round(float(items[4])))
    print("pared rect {0},{1},{2},{3}".format(x,y,w,h))
    return x,y,w,h


import signal, errno
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        pass

    orig_handler = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, orig_handler)



# 
# We create the tile in a temp directory and then move it to its final
# destination.
#
def writeTilesFromDetects(tileDir, detects, origFile):
    if detects == None or len(detects) == 0:
        return
    tempTileDir = tempfile.mkdtemp(dir='/home/trbatcha/tempDir')
    os.chmod(tempTileDir, 0o777)

    outputDir = os.path.join(tempTileDir, "output")
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        os.chmod(outputDir, 0o777)
    vname = os.path.basename(origFile)
    vPath = os.path.join(tempTileDir, vname + ".vrt")
    listPath = os.path.join(tempTileDir, "fileList.txt")
    listFile = open(listPath, "w")
    basename = os.path.basename(origFile)
    nname, ext = os.path.splitext(origFile)
    if ext.lower() == '.tif':
        bitSize = 8
    else:
        bitSize = 16
    for detect in detects:
        print(detect)
        rectStr = detect
        x, y, w, h = parseRectStr(rectStr)
        print("detect = {0},{1},{2},{3}".format(x, y,w , h))
        tName = basename + "_" + str(x) + "_" + str(y) + "_" + \
            str(w) + "_" + str(h) + ".tif"
        t2Name = basename + "_" + str(x) + "_" + str(y) + "_" + \
            str(w) + "_" + str(h) + "_w" + ".tif"
        tPath = os.path.join(tempTileDir, tName)
        t2Path = os.path.join(tempTileDir, t2Name)
        if os.path.exists(tPath) == True:
            os.remove(tPath)
        if os.path.exists(t2Path) == True:
            os.remove(t2Path)
        # Git the image clip
        if bitSize == 16:
            transStr = "/home/trbatcha/tools/bin/gdal_translate -of GTiff " +\
             "-ot Byte -scale 64 1024 0 255 -b 1 -srcwin " \
             + str(x) + " " +  str(y) + " " + str(w) + " " + str(h) + " " \
             + origFile + " " + tPath
        else:
            transStr = "/home/trbatcha/tools/bin/gdal_translate -of GTiff " +\
             "-ot Byte -b 1 -srcwin " \
             + str(x) + " " +  str(y) + " " + str(w) + " " + str(h) + " " \
             + origFile + " " + tPath
        args = shlex.split(transStr)
        print("running translate")
        p = Popen(args, stdout=PIPE, stderr=PIPE)
        pstdout, pstderr = p.communicate()
        print pstderr
        print("translate complete")
        #get rid of xml file gdal_translate creates
        xmlfile = tPath + ".aux.xml"
        if os.path.exists(xmlfile):
            os.remove(tPath + ".aux.xml")
        print (pstdout)
        warpStr = "/home/trbatcha/tools/bin/gdalwarp -of GTiff -t_srs " + \
                  "EPSG:3857  -overwrite "  + tPath + " " + t2Path
        args = shlex.split(warpStr)
        print("running warp")
        p = Popen(args, stdout=PIPE, stderr=PIPE)
        pstdout, pstderr = p.communicate()
        print (pstderr)
        print (pstdout)
        print("warp complete")
        listFile.write(t2Path + '\n')

    listFile.close()
    vrtStr = "/home/trbatcha/tools/bin/gdalbuildvrt -srcnodata 0 -addalpha " \
             + "-vrtnodata 0 -overwrite -input_file_list " + listPath + \
             " " + vPath
    args = shlex.split(vrtStr)
    print("running vrt")
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    pstdout, pstderr = p.communicate()
    print (pstderr)
    print (pstdout)
    print("virt complete")
    
    # Generate tiles for all the image chips
    import gdal2tiles
    tileStr = "-v -p mercator --zoom '13-16' -w none " + vPath + " " + outputDir
    #debug tried gdal2tiles as seperate process, it did not fix my problem so commented out
    #my_env = os.environ.copy()
    #tileStr = "/home/trbatcha/tools/bin/python gdal2tiles.py -v -p mercator -z 13 -w none " + vPath + " " + outputDir
    args = shlex.split(tileStr)
    # run it in seperate shell for clean up
    #p = Popen(args, env=my_env, stdout=PIPE, stderr=PIPE)
    #stdout, stderr = p.communicate()
    #print (stderr)
    #print (stdout)

    print("gen tiles")

    tileGenFailed = False
    with timeout(10):
        try:
            # By default gdal turns exceptions off
            gdal.UseExceptions()
            targs = gdal.GeneralCmdLineProcessor(args)
            gtiles = gdal2tiles.GDAL2Tiles(targs)
            gtiles.process()
        except IOError, err:
            if err.errno != errno.EINTR:
                print("gdal2tiles FAILED!!!")
                print(err)
                sys.stdout.flush()
                shutil.rmtree(tempTileDir, ignore_errors=True)
                return
            print("TileGeneration TIMED OUT!! for file " + origFile) 
            tileGenFailed = True

    print("gen tiles complete")
    # before we move tiles lets check lockfile and wait if not avail
    if tileGenFailed == False:
        with timeout(3):
            lockFile = os.path.join(tileDir,"tileLock")
            lock = open(lockFile, 'w+')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
            except IOError, e:
                if e.errno != errno.EINTR:
                    raise e
                print("Tile filelock timeout")
                lock.close()
                shutil.rmtree(tempTileDir, ignore_errors=True)
                return
        moveTiles(outputDir, tileDir)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
     
    # remove the non-tiles we created
    shutil.rmtree(tempTileDir, ignore_errors=True)




if __name__ == '__main__':

    os.umask(0)
    #debug force stdout to flush optut
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 0) 
    print("Running doTile.py")

    args = parse_args()

    tiledir = args.tile_path
    dstring = args.dstring
    print("Recieved dstring: " + dstring)

    tempDir = tempfile.mkdtemp(dir = "/dev/shm")
    os.chmod(tempDir, 0o777)


    # If the string is surrounded my double quotes remove them
    if dstring[0] == dstring[-1] and dstring.startswith(("'", '"')):
        dstring = dstring[1:-1]
    origFile, type, x, y, width, height  = dstring.split(" ")
    detectList = []
    
    detectList.append(type + " " + x + " " + y + " " + width + " " + height)
    print(detectList[0])
    temp = detectList[0].split(' ')
    print(temp[4])

    writeTilesFromDetects(tiledir, detectList, origFile)

    shutil.rmtree(tempDir)

    sys.exit(0)

