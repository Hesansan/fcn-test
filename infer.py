import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
import vis
import caffe
 
import caffe
import glob
fileList=glob.glob('.../data/JPEGImages/TEST/*.jpg')                                
#val = np.loadtxt('.../data/ImageSets/Segmentation/seg11valid11.txt', dtype=str)
for filename in fileList:
   
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(filename)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    # init
    #caffe.set_device(int(sys.argv[1]))
    caffe.set_device(2)
    caffe.set_mode_gpu()
    # load net
    
    net=caffe.Net('deploy.prototxt', 'short_iter_62000.caffemodel', caffe.TEST)
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0) 
    voc_palette = vis.make_palette(21)
    out_im = Image.fromarray(vis.color_seg(out, voc_palette))
    target_dir = filename[:-4].rsplit('/',1)[0]
    jpg_name = filename.rsplit('/',1)[1].split('.')[0]
    out_im.save(target_dir+'/TEST/'+jpg_name+'.png')
    #plt.imshow(out,cmap='gray')
    #plt.axis('off')
    #plt.savefig(filename)
    #plt.show()filename[:-4]
    # np.save(filename[:-4], out)