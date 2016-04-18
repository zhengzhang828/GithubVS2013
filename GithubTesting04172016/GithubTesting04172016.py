"""Preprocessing script.

This script walks over the directories and dump the frames into a csv file
"""
import os
import csv
import sys
import random
import scipy
import numpy as np
import dicom
import cv2
import glob
import math
from collections import defaultdict
import pickle
import gzip
#from skimage import data, color, img_as_ubyte
from skimage.restoration import denoise_bilateral
 

from skimage import transform

def mkdir(fname):
   try:
       os.mkdir(fname)
   except:
       pass

def save(object, filename, bin = 1):
	"""Saves a compressed object to disk
	"""
	file = gzip.GzipFile(filename, 'wb')
	file.write(pickle.dumps(object, bin))
	file.close()

#whay
def load(filename):
	"""Loads a compressed object from disk
	"""
	file = gzip.GzipFile(filename, 'rb')
	buffer = ""
	while 1:
		data = file.read()
		if data == "":
			break
		buffer += data
	object = pickle.loads(buffer)
	file.close()
	return object 

def get_frames(root_path):
   """Get path to all the frame in view SAX and contain complete frames"""
   print(root_path)
   counter = 0
   counter2 = 0
   cprefix = ""
   ret = []
   for root, _, files in os.walk(root_path):
       
       #if len(files) == 0 or not files[0].endswith(".dcm") or root.find("sax") == -1:
       if len(files) == 0 or not files[0].endswith(".dcm"):
           continue

       #if  counter == 5:
       #    cprefix = files[0].rsplit('-', 1)[0]
       #    counter = 0
       #    continue
       #if  cprefix == files[0].rsplit('-', 1)[0]:
       #    continue

       prefix = files[0].rsplit('-', 1)[0]
       fileset = set(files)
       expected = ["%s-%04d.dcm" % (prefix, i + 1) for i in range(30)]
       if all(x in fileset for x in expected):
           ret.append([root + "/" + x for x in expected])
           counter += 1
           counter2 += 1
           sys.stdout.write( str(counter2)+" ")
   # sort for reproduciblity
   return sorted(ret, key = lambda x: x[0])


def get_label_map(fname):
   labelmap = {}
   fi = open(fname)
   fi.readline()
   for line in fi:
       arr = line.split(',')
       labelmap[int(arr[0])] = line
   return labelmap


def write_label_csv(fname, frames, label_map):
   fo = open(fname, "w")
   for lst in frames:
       print(lst[0])
       index = int(lst[0].split("/")[2])
       if label_map != None:
           fo.write(label_map[index])
       else:
           fo.write("%d,0,0\n" % index)
   fo.close()

PatientSex ={}
PatientAge ={}
SliceLocation ={}

show_images             = False
show_circles            = False
show_combined_centers   = False
show_main_center        = False
center_distance_devider = 3
best_cluster_divider    = 25.0
contour_roundness       = 2.5
black_removal           = 200
max_area_devider        = 12
minimal_median          = 0

def biggest_cluster((xx, yy), centers, relax_mult=1):
    image_center = (float(xx) / 2, float(yy) / 2)
    max_kernel_radius = float(min(xx, yy)) / best_cluster_divider * relax_mult
    neighbours = [0 for x in range(len(centers))]
    for i_i, i in enumerate(centers):
        for i_j, j in enumerate(centers):
            if i_i == i_j:
                continue
            x1 = float(i[0])
            x2 = float(j[0])
            y1 = float(i[1])
            y2 = float(j[1])
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance < max_kernel_radius:
                neighbours[i_i] += 1
    # finding biggest cluster
    max_distance = float(min(xx, yy)) / center_distance_devider
    checked = 0
    while checked < len(centers):
        if max(neighbours) == 0:
            checked = len(centers) + 1
        index = neighbours.index(max(neighbours))
        x1 = float(centers[index][0])
        y1 = float(centers[index][1])
        distance = math.hypot(image_center[0] - x1, image_center[1] - y1)
        #print distance, max_distance
        if distance > max_distance:
            neighbours[index] = 0
        else:
            checked = len(centers) + 1
    #print max_distance, max_kernel_radius, xx, yy, neighbours.index(max(neighbours))
    #print neighbours
    return centers[neighbours.index(max(neighbours))]

def get_circles(image):
    v = np.median(image)
    upper = int(min(255, (1.0 + 5) * v))
    print("upper {}".format(upper))
    i = 40
    while True:
        circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,50,
                                param1=upper,param2=i,minRadius=0,maxRadius=40)
        i -= 1
        if circles is None:
            pass
        else:
            circles = np.uint16(np.around(circles))
            break
    return circles


def write_data_csv(fname, frames, preproc):
   """Write data to csv file"""
   #fdata = open(fname, "w")
   #dwriter = csv.writer(fdata)
   counter = 0
   result = []
   
   for lst in frames:
       data = []
       imglist = []
       circlesall = []
       for path in lst:
           #try:
               #dst_path = "../"+path.rsplit(".", 1)[0] + ".64x64.jpg"
               #dst_path = dst_path.replace("./train","kaggleimgdatafinal").replace("./validate","kagglevimgdatafinal").replace("./test","kaggletimgdatafinal")
               #print(dst_path)
               #if os.path.exists( dst_path):
               #    print("skip")
               #    continue
               f = dicom.read_file(path)
               (PixelSpacingx, PixelSpacingy) = f.PixelSpacing
               (PixelSpacingx, PixelSpacingy) = (float(PixelSpacingx), float(PixelSpacingy))
               img =  f.pixel_array.astype('uint8')  
               img = cv2.equalizeHist(img)
               imglist.append(crop_resize_other(img,PixelSpacingx))
           #except:
           #    print(sys.exc_info()[0])

       for img in imglist:
          cir = get_circles(img) 
          circlesall.append(cir)
       centers = []
       for i in circlesall:
            if i is None:
                continue
            for c in i[0,:]:
                centers.append([c[0],c[1]])
       print ("Looking for biggest_cluster {}".format(len(centers)))
       (xx, yy) = (None, None)
       (cx, cy) = (None, None)
       if len( imglist) > 0:
           (xx, yy) = imglist[0].shape
           (cx, cy) = biggest_cluster((xx, yy), centers)
           print("center {}".format((cx, cy) ))
       else:
           continue

       for path in lst:
           try:
               dst_path = "../"+path.rsplit(".", 1)[0] + ".64x64.jpg"
               dst_path = dst_path.replace("./train","kaggleimgdatafinal").replace("./validate","kagglevimgdatafinal").replace("./test","kaggletimgdatafinal")
               print(dst_path)
               #if os.path.exists( dst_path):
               #    print("skip")
               #    continue
               f = dicom.read_file(path,force=True)
 
               (PixelSpacingx, PixelSpacingy) = f.PixelSpacing
               (PixelSpacingx, PixelSpacingy) = (float(PixelSpacingx), float(PixelSpacingy))
               img = preproc(f.pixel_array.astype(float) / np.max(f.pixel_array),  PixelSpacingx, (cx, cy))

               print(os.path.dirname(dst_path))
               if not os.path.exists(os.path.dirname(dst_path)):
                    os.makedirs(os.path.dirname(dst_path))
               scipy.misc.imsave(dst_path, img)
           except:
               print(sys.exc_info()[0])
 
       counter += 1
       if counter % 100 == 0:
           print("%d slices processed" % counter)
   print("All finished, %d slices in total" % counter)
   #fdata.close()
   return result

def auto_canny(image, sigma=0.05):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, 100, 250)
 
	# return the edged image
	return edged

def crop_resize_other(img, pixelspacing ):
   print("image shape {}".format(np.array(img).shape))

   xmeanspacing = float(1.25826490244)
   ymeanspacing = float(1.25826490244)

   xscale = float(pixelspacing) / xmeanspacing
   yscale = float(pixelspacing) / ymeanspacing
   xnewdim = round( xscale * np.array(img).shape[0])
   ynewdim = round( yscale * np.array(img).shape[1])
   img = transform.resize(img, (xnewdim, ynewdim))
  
   #img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) 
   #img = auto_canny(img)
   img = denoise_bilateral(img, sigma_range=0.05, sigma_spatial=15)

   #im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX) 
   #img = img.Canny(im,128,128)
   """crop center and resize"""
   if img.shape[0] < img.shape[1]:
       img = img.T
   # we crop image from center
   short_egde = min(img.shape[:2])
   yy = int((img.shape[0] - short_egde) / 2)
   xx = int((img.shape[1] - short_egde) / 2)
   crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
   crop_img *= 255
   return crop_img.astype("uint8")

def crop_resize(img, pixelspacing, center, size):
   print("image shape {}".format(np.array(img).shape))

   xmeanspacing = float(1.25826490244)
   ymeanspacing = float(1.25826490244)

   xscale = float(pixelspacing) / xmeanspacing
   yscale = float(pixelspacing) / ymeanspacing
   xnewdim = round( xscale * np.array(img).shape[0])
   ynewdim = round( yscale * np.array(img).shape[1])
   img = transform.resize(img, (xnewdim, ynewdim))
   img = np.uint8(img * 255)
   #img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) 
   #img = auto_canny(img)
   img = denoise_bilateral(img, sigma_range=0.05, sigma_spatial=15)

   #im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX) 
   #img = img.Canny(im,128,128)
   """crop center and resize"""
   if img.shape[0] < img.shape[1]:
       img = img.T
   # we crop image from center
   short_egde = min(img.shape[:2])
   #yy = int((img.shape[0] - short_egde) / 2)
   #xx = int((img.shape[1] - short_egde) / 2)
   #crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
   # resize to 64, 64
   resized_img = None
   yy = int(center[1] - 64)
   xx = int(center[0] - 64)
   if yy > 0 and xx > 0 and (yy+128) < img.shape[0] and xx + 128 < img.shape[1]:
       resized_img = img[yy : yy + 128, xx : xx + 128]
   else:
       yy = int((img.shape[0] - short_egde) / 2)
       xx = int((img.shape[1] - short_egde) / 2)
       resized_img = img[yy : yy + 128, xx : xx + 128]

   resized_img *= 255
   return resized_img.astype("uint8")


def local_split(train_index):
   random.seed(0)
   train_index = set(train_index)
   all_index = sorted(train_index)
   num_test = int(len(all_index) / 3)
   random.shuffle(all_index)
   train_set = set(all_index[num_test:])
   test_set = set(all_index[:num_test])
   return train_set, test_set


def split_csv(src_csv, split_to_train, train_csv, test_csv):
   ftrain = open(train_csv, "w")
   ftest = open(test_csv, "w")
   cnt = 0
   for l in open(src_csv):
       if split_to_train[cnt]:
           ftrain.write(l)
       else:
           ftest.write(l)
       cnt = cnt + 1
   ftrain.close()
   ftest.close()

# Load the list of all the training frames, and shuffle them
# Shuffle the training frames
random.seed(10)
train_frames = get_frames("./train")
#random.shuffle(train_frames)
validate_frames = get_frames("./validate")

test_frames = get_frames("./test")

# Write the corresponding label information of each frame into file.
write_label_csv("./train-label.csv", train_frames, get_label_map("./train.csv"))
write_label_csv("./validate-label.csv", validate_frames, None)

# Dump the data of each frame into a CSV file, apply crop to 64 preprocessor
train_lst = write_data_csv("./train-64x64-data.csv", train_frames, lambda x,y,z: crop_resize(x,y,z, 128))
valid_lst = write_data_csv("./validate-64x64-data.csv", validate_frames, lambda x,y,z: crop_resize(x,y,z, 128))
test_lst = write_data_csv("./test-64x64-data.csv", test_frames, lambda x,y,z: crop_resize(x,y,z, 128))

# Generate local train/test split, which you could use to tune your model locally.
#train_index = np.loadtxt("./train-label.csv", delimiter=",")[:,0].astype("int")
#train_set, test_set = local_split(train_index)
#split_to_train = [x in train_set for x in train_index]
#split_csv("./train-label.csv", split_to_train, "./local_train-label.csv", "./local_test-label.csv")
#split_csv("./train-64x64-data.csv", split_to_train, "./local_train-64x64-data.csv", "./local_test-64x64-data.csv")

#metadata= {'PatientSex': PatientSex, 'PatientAge': PatientAge , 'SliceLocation': SliceLocation  }   
#save(metadata, 'metadata.pkl')
