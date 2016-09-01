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

#import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from contextlib import contextmanager
import shutil
from subprocess import Popen, PIPE
import shlex
import tempfile
import re
from timeit import default_timer as timer

from osgeo import gdal

CLASSES = ('__background__',
           'ship')
           #'fast_ship')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'vgg': ('VGG_CNN_M_1024',
                   'VGG_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

def merged_stderr_stdout():
    return stdout_redirected(to=sys.stdout, stdout=sys.stderr)

def merged_stdout_stderr():
    return stdout_redirected(to=sys.stderr, stdout=sys.stdout)

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, im_file):
    """Detect object classes in an image using pre-computed object proposals."""
    # compute the file offset from the name
    iterEx = re.compile(".*?_(\d+)_(\d+)\.jpg$")
    itIter = iterEx.findall(im_file)
    if len(itIter) > 0:
        xoff = int(itIter[0][0])
        yoff = int(itIter[0][1])
    else:
        print("Bad Filename " + im_file + ".  No offsets! Skipping file")
        return []
    # Load the demo image as gray scale
    gim = cv2.imread(im_file, flags= cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # convert to rgb repeated in each channel
    im = cv2.cvtColor(gim, cv2.COLOR_GRAY2BGR)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    res = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:,-1] >= CONF_THRESH)[0]
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            res.append(cls + " {0} {1} {2} {3}".format(xoff + bbox[0], yoff + bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

    return res

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='vgg')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--model', dest='model_file',
                        help='caffe model file',
                        default=None, type=str)
    parser.add_argument('--proto', dest='proto_file',
                        help='caffe prototext file',
                        default=None, type=str)
    parser.add_argument('--split', dest='split_size',
                        help='width && height for split up images',
                        action='store', type=int)
    parser.add_argument('--tiles', dest='tile_path',
                        help='image tile output path',
                        default=None, type=str)
    parser.add_argument('file', help="Image file or dir to process",
                        type=str)

    args = parser.parse_args()

    return args

def split_up_file(fname, tempDir, splitSize, maxCnt):

    dset = gdal.Open(fname)
    width = dset.RasterXSize
    height = dset.RasterYSize
    baseName = os.path.basename(fname)
    tName = os.path.join(tempDir, baseName)
    fileList = []
    cnt = 1
    start = timer()
    print("starting splits")
    sys.stdout.flush()
    for i in range(0, width, splitSize):
        for j in range(0, height, splitSize):
            if maxCnt > 0 and cnt > maxCnt:
                break
            cnt += 1
            w = min(i+splitSize, width) - i
            h = min(j+splitSize, height) - j
            xoff = i
            yoff = j
            if w < splitSize:
                xoff = i - (splitSize - w)
                if xoff < 0:
                    xoff = 0
            if h < splitSize:
                yoff = j - (splitSize - h)
                if yoff < 0:
                    yoff = 0
            tempName = tName + "_" + str(i) + "_" + str(j) + ".jpg"
            fileList.append(tempName)

            transStr = "/home/trbatcha/tools/bin/gdal_translate -of JPEG -ot Byte -scale 64 1024 0 255 -b 1 -srcwin " + str(xoff) + " " + str(yoff) + \
                   " " + str(splitSize) + " " + str(splitSize) + " " + fname + " " + tempName
            #result = subprocess.check_output([transStr], shell=True)
            args = shlex.split(transStr)
            p = Popen(args, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
            print stderr
            #get rid of xml file gdal_translate creates
            xmlfile = tempName + ".aux.xml"
            if os.path.exists(xmlfile):
                os.remove(tempName + ".aux.xml")
            print (stdout)
            sys.stdout.flush()
    end = timer()
    print ("Split time = {0}".format(end - start))
    sys.stdout.flush()
    return fileList

def doWriteToHDFS(dirname, fname) :
    basename = os.path.basename(fname)
    hname = os.path.join(dirname, basename)
    put = Popen(["hdfs", "dfs", "-put", fname, hname],
            stdout=PIPE, stderr = PIPE)
    stdout, stderr = put.communicate()
    print stderr
    return hname

def mergeTiles(src, dst):
    img1 = mpimg.imread(src)
    img2 = mpimg.imread(dst)
    img = cv2.bitwise_or(img1, img2)
    mpimg.imsave(dst, img)

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
            if os.path.exists(dname) == True and ext == '.png':
                mergeTiles(sname, dname)
            else:
                shutil.move(sname, dname)
            os.chmod(dname, 0o666)

def parseRectStr(rectStr):
    items = rectStr.split(' ')
    # the first item is the class which we will ignore
    x = int(round(float(items[1])))
    y = int(round(float(items[2])))
    w = int(round(float(items[3])))
    h = int(round(float(items[4])))
    return x,y,w,h

# 
# We create the tile in a temp directory and then move it to its final
# destination.
#
def writeTilesFromDetects(tileDir, detects):
    tempTileDir = "/home/trbatcha/tempDir/temptiles"
    outputDir = os.path.join(tempTileDir, "output")
    if not os.path.exists(tempTileDir):
        os.makedirs(tempTileDir)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    vPath = os.path.join(tempTileDir, "vFile.vrt")
    listPath = os.path.join(tempTileDir, "fileList.txt")
    listFile = open(listPath, "w")
    for detect in detects:
        rectStr = detect[0]
        imgName = detect[1]
        basename = os.path.basename(imgName)
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
        transStr = "/home/trbatcha/tools/bin/gdal_translate -of GTiff " +\
         "-ot Byte -scale 64 1024 0 255 -b 1 -srcwin " \
         + str(x) + " " +  str(y) + " " + str(w) + " " + str(h) + " " \
         + imgName + " " + tPath
        args = shlex.split(transStr)
        print("running translate")
        p = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        print stderr
        print("translate complete")
        #get rid of xml file gdal_translate creates
        xmlfile = tPath + ".aux.xml"
        if os.path.exists(xmlfile):
            os.remove(tPath + ".aux.xml")
        print (stdout)
        warpStr = "/home/trbatcha/tools/bin/gdalwarp -of GTiff -t_srs " + \
                  "EPSG:3857  -overwrite "  + tPath + " " + t2Path
        args = shlex.split(warpStr)
        print("running warp")
        p = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        print (stderr)
        print (stdout)
        print("warp complete")
        listFile.write(t2Path + '\n')

    listFile.close()
    vrtStr = "/home/trbatcha/tools/bin/gdalbuildvrt -srcnodata 0 -addalpha " \
             + "-vrtnodata 0 -overwrite -input_file_list " + listPath + \
             " " + vPath
    args = shlex.split(vrtStr)
    print("running vrt")
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    print (stderr)
    print (stdout)
    print("virt complete")
    
    # Generate tiles for all the image chips
    import gdal2tiles
    tileStr = "-v -p mercator -z 13 -w none " + vPath + " " + outputDir
    args = shlex.split(tileStr)
    print("gen tiles")
    targs = gdal.GeneralCmdLineProcessor(args)
    gtiles = gdal2tiles.GDAL2Tiles(targs)
    gtiles.process()
    print("gen tiles complete")
    moveTiles(outputDir, tileDir)

    # remove the non-tiles we created
    shutil.rmtree(tempTileDir, ignore_errors=True)




if __name__ == '__main__':
    save_stdout = sys.stdout
    sys.stdout = sys.stderr
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    if args.cfg_file is not None:
        print("using config " + args.cfg_file)
        cfg_from_file(args.cfg_file)
    if cfg.TRAIN.IS_COLOR == True:
        print("We are configured for color")
    else:
        print("We are configured for b/w")
    if args.split_size:
        print("We are to split up image by {0}".format(args.split_size))
    else:
        print("No split applied.")
    tiledir = args.tile_path
    ifile = args.file
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'VGG_CNN_M_1024',
                            'faster_rcnn_end2end', 'test_ships.prototxt')
    #caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'faster_rcnn_end2end',
    #                         'ak47_train', 'zf_faster_rcnn_iter_70000.caffemodel')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'faster_rcnn_end2end',
                             'ships_train', 'vgg_cnn_m_1024_faster_rcnn_iter_500000.caffemodel')
    if args.model_file is not None:
        caffemodel = args.model_file
    if args.proto_file is not None:
        prototxt = args.proto_file

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you train it?'
                       ).format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    # Need to redirect stdout since we only want to return the results
    tempDir = tempfile.mkdtemp(dir = "/dev/shm")

    #debug
    doDetect = True
    detects = []


    if args.split_size != None:
        #fileList = split_up_file(ifile, tempDir, args.split_size, -1)
        ##debug only do the first one
        fileList = split_up_file(ifile, tempDir, args.split_size, 1)
    else:
        fileList = [ifile]


    if doDetect == True:
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        print '\n\nLoaded network {:s}'.format(caffemodel)
        # Warmup on a dummy image
        im = 128 * np.ones((300, 500, 1), dtype=np.uint8)
        #im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _= im_detect(net, im)

        for nextf in fileList:
            print('detection for ' + nextf)
            res = demo(net, nextf)
            if res != None and len(res) > 0:
                print("we have detection results!")
                for d in res:
                    #We have to use ifile instead of nextf since
                    #all the geo data is gone from nextf
                    detects.append((d, ifile))
            else:
                #For testing
                print("appending dummp detection")
                detects.append(("ship 12300.0 11600.0 500.0 500.0", ifile))



    #debug show all detects
    print("Printing detects for {0} detections".format(len(detects)))
    for d in detects:
        print(d[0])

    writeTilesFromDetects(tiledir, detects)
    shutil.rmtree(tempDir)

    #putting stdout back so we can output the results
    sys.stdout = save_stdout
    # Write out the result to stdout
    for d in detects:
        print(d[0])
    sys.exit(0)

