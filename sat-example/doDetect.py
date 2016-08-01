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
    for i in range(0, width, splitSize):
        for j in range(0, height, splitSize):
            if cnt > maxCnt:
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
    end = timer()
    print ("Split time = {0}".format(end - start))
    return fileList


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
            if res != None:
                detects.extend(res)

    shutil.rmtree(tempDir)
    sys.stdout = save_stdout

    for d in detects:
        print(d)

    sys.exit(0)

