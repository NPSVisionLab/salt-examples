from setuptools import setup, find_packages
setup (name = 'py-faster-rcnn',
       version = "0.0.16",
       package_dir = {'':'.'},
       packages = ['','datasets','fast_rcnn','nms','roi_data_layer','rpn','transform', 'utils', 'caffe', 'caffe.proto'],
       package_data = { 'nms':['cpu_nms.so', 'gpu_nms.so'],
                       'utils':['cython_bbox.so'],
                       'caffe':['_caffe.so'],
                      },
       include_package_data = True
      )
