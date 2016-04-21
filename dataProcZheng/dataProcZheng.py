import os
import pickle
import gzip
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
import pylab
from skimage.restoration import denoise_bilateral
from skimage import transform
import pylab
import cv2
import matplotlib.pyplot as plt


PatientSex = {}
PatientAge = {}
SliceLocation = {}

show_images = False
show_circles = False
show_combined_centers = False
show_main_center = False
center_distance_devider = 3
best_cluster_divider = 25.0
contour_roundness = 2.5
black_removal = 200
max_area_devider = 12
minimal_median = 0


#Create folder in the system
def mkdir(fname):
    try:
        os.mkdir(fname)
    except:
        pass

def save(object, filename, bin=1):
    tempfile = gzip.GzipFile(filename,"wb")
    try:
        tempfile.write(pickle.dumps(object,bin))
    finally:
        tempfile.close()

def get_circles(image):
    v = np.median(image)
    upper = int(min(255, (1.0 + 5) * v))
    #Notice attention
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

def get_circles_zz(image):
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

def get_frames(root_path):
    i = 0
    t = 0
    counter = 0
    counter2 = 0
    filesColl = []
    ret = []
    for root, _, files in os.walk(root_path):
        if len(files) == 0 or not files[0].endswith(".dcm"):
                continue
        prefix = files[0].rsplit('-',1)[0] #Take the patient name
        #set object of constructing and manipulating unordered collections of unique elements
        fileset = set(files) 
        expected =["%s-%04d.dcm" % (prefix,i + 1) for i in range(30)]
        if all(x in fileset for x in expected):
            ret.append([root + "/" + x for x in expected])
            counter += 1
            counter2 += 1
    return sorted(ret, key = lambda x: x[0])

def get_label_map(fname):
    labelmap = {}
    fi = open(fname)
    fi.readline()
    for line in fi:
        arr = line.split(',')
        labelmap[int(arr[0])] = line
    return labelmap

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

#find the biggest cluster, biggest cluster refers to the point that has the most circles
def biggest_cluster_zz((xx,yy),centers,relax_mult=1):
    image_center = (float(xx) / 2, float(yy) / 2)
    max_kernel_radius = float(min(xx,yy)) / best_cluster_divider*relax_mult
    neighbours = [0 for x in range(len(centers))] # neighbours has the same length as the centers of the image, same number of circles
    for i_i, i in enumerate(centers):
        for i_j, j in enumerate(centers):
            if i_i == i_j:
                continue
            x1 = float(i[0])
            x2 = float(j[0])
            y1 = float(i[0])
            y2 = float(j[1])
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance < max_kernel_radius:
                neighbours[i_i] += 1#collect center coordinates of the circle which the two center distance smaller then max_kernel_radius
                
    #find biggest cluster
    max_distance = float(min(xx,yy)) / center_distance_devider #divide the image into 3 part and get the distance
    checked = 0
    while checked < len(centers):
        if max(neighbours) == 0:
            checked = len(centers) + 1
        index = neighbours.index(max(neighbours)) # find the center which has the most neighbour, and return the index of this center point
        x1 = float(centers[index][0]) # according to the index, find out the center coordinates
        y1 = float(centers[index][1])
        distance = math.hypot(image_center[0] - x1,image_center[1] - y1)#calculate the eculadian distance of the two points
        if distance > max_distance:
            neighbours[index] = 0
        else:
            checked = len(centers) + 1
    cp = centers[neighbours.index(max(neighbours))]
    return cp

#print label_map
def write_label_csv(fname, frames, label_map):
    fo = open(fname, "w")
    for lst in frames:
        #print(lst[0])
        print "lst[0]", lst[0]
        #-------------------split the root and extract the folder number------------------
        index = int(lst[0].split("\\")[6])
        #print label_map[index]
        if label_map != None:
            fo.write(label_map[index])
        else:
            fo.write("%d,0,0\n" % index)
    fo.close()

def crop_resize_other(img, pixelspacing ):
   
   print("image shape {}".format(np.array(img).shape))

   xmeanspacing = float(1.25826490244)
   ymeanspacing = float(1.25826490244)

   xscale = float(pixelspacing)/xmeanspacing #pixelspacing[0]
   yscale = float(pixelspacing)/ymeanspacing #pixelspacing[1]
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
   #Question, notice, attention, yy, xx, why shape 0 is y? should shape 0 is xx?

   yy = int((img.shape[0] - short_egde) / 2)
   xx = int((img.shape[1] - short_egde) / 2)
   crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
   crop_img *= 255
   return crop_img.astype("uint8")

def crop_resize_other_1(img, pixelspacing):#normalize image
        #-------------------------------------
        #thresholdval = 20
        #r,g,b = img.splitChannels()
        #img = g.equalize().threshold(thresholdval).invert()
        #img.show()
        #---------------

        xmeanspacing = 1.25826490244
        ymeanspacing = 1.25826490244

        #pixelspacing = (PixelSpacingx, PixelSpacingy)
        xmeanspacing = float(xmeanspacing)
        ymeanspacing = float(ymeanspacing)
        xscale = float(pixelspacing[0])/xmeanspacing
        yscale = float(pixelspacing[1])/ymeanspacing
        xnewdim = round(xscale*np.array(img).shape[0])
        ynewdim = round(yscale*np.array(img).shape[1])
        img = transform.resize(img,(xnewdim, ynewdim))
        img = np.uint8(img*255)

        #img = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15,multichannel=False)
        #img = denoise_bilateral(img,sigma_range=0.05,multichannel=False)
        if img.shape[0] < img.shape[1]:
            img = img.T
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy:yy + short_edge,xx:xx + short_edge]
        crop_img *= 255
       
        return crop_img.astype("uint8")
        
def crop_resize(img, pixelspacing, center, size):

            #print 'img, pixelspacing, center, size', (img, pixelspacing, center, size)
    print("image shape {}".format(np.array(img).shape))
    xmeanspacing = float(1.25826490244)
    ymeanspacing = float(1.25826490244)

    xscale = float(pixelspacing)/xmeanspacing
    yscale = float(pixelspacing)/ymeanspacing

    #temp = xscale*np.array(img)
    xnewdim = round(xscale*np.array(img).shape[0])
    ynewdim = round(yscale*np.array(img).shape[1])

    img = transform.resize(img, (xnewdim, ynewdim))
    img = np.uint8(img*255)


    if img.shape[0] < img.shape[1]:
        img = img.T

    short_edge = min(img.shape[:2])
    resized_img = None

    yy = int(center[1] - 64)
    xx = int(center[0] - 64)
    if yy > 0 and xx > 0 and (yy + 128) < img.shape[0] and xx + 128 < img.shape[1]:
        resized_img = img[yy : yy + 128, xx : xx + 128]
    else:
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        resized_img = img[yy : yy + 128, xx : xx + 128]

    resized_img *= 255
    img = resized_img.astype("uint8")
    return img

def write_data_csv(frames, preproc):
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
               ##Questions PixelSpacingx, PixelSpacingy should have y as well?
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
                dst_path = path.rsplit(".",1)[0]+".64x64.jpg"
                print "place 1"
                print "dst_path: ",dst_path
                dst_path = dst_path.replace("train","kaggleimgdatafinal")
                #-------AWS path----------
                #dst_path = dst_path.replace("train","kaggleimgdatafinal").replace("validate","kagglevimgdatafinal").replace("test","kaggletimgdatafinal")
                #--------------------------
                print "place 2"
                #dst_path = "../" + path.rsplit(".",1)[0]+".64x64.jpg"
                #dst_path = dst_path.replace("./train","kaggleimgdatafinal").replace("./validate","kagglevimgdatafinal").replace("./test","kaggletimgdatafinal")
                
                #print "dst_path: ",dst_path
                f = dicom.read_file(path,force=True)
                (PixelSpacingx,PixelSpacingy) = f.PixelSpacing
                (PixelSpacingx,PixelSpacingy) = (float(PixelSpacingx),float(PixelSpacingy))
            
                img = f.pixel_array.astype(float) / np.max(f.pixel_array)
                pixelspacing = PixelSpacingx
                center = (cx,cy)
                img = preproc(img, pixelspacing,center)
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

def write_data_csv_1(frames, preproc):
    counter = 0
    result = [] 
    for lst in frames:
        data = []
        imglist = []
        circlesall = []
        for path in lst:
            f = dicom.read_file(path)
            (PixelSpacingx, PixelSpacingy) = f.PixelSpacing
            (PixelSpacingx, PixelSpacingy) = (float(PixelSpacingx), float(PixelSpacingy))
            pixelspacing = (PixelSpacingx, PixelSpacingy)    
            img = f.pixel_array.astype('uint8')
            #print f.PixelSpacing
            #print 'img: ',img
            #cv2.imshow('img',img)
            #cv2.waitKey()
            img = cv2.equalizeHist(img)
            imglist.append(crop_resize_other(img,pixelspacing))
            #print "working"
            #cv2.imshow('img',img)
            #cv2.waitKey()
            #print imglist
        for img in imglist:
            cir = get_circles(img)
            circlesall.append(cir)
            #Attention remove break
            #break

        centers = []
        
        for i in circlesall: #i returns the center location coordinates and the radius i[0], i[1], i[2]
            if i is None:
                continue
            for c in i[0,:]:
                centers.append([c[0],c[1]])
        print ("Looking for biggest_cluster {}".format(len(centers)))
        (xx,yy) = (None,None)
        (cx,cy) = (None,None)
    
        relax_mult = 1
        if len(imglist)>0:
            (xx,yy) = imglist[0].shape
            #biggest cluster
            (cx,cy) = biggest_cluster((xx,yy),centers)
            print("center {}".format((cx,cy)))
        else:
            continue
    
        #print "lst: ",lst
        for path in lst:
            try:
                dst_path = path.rsplit(".",1)[0]+".64x64.jpg"
                print "place 1"
                print "dst_path: ",dst_path
                dst_path = dst_path.replace("train","kaggleimgdatafinal")
                #-------AWS path----------
                #dst_path = dst_path.replace("train","kaggleimgdatafinal").replace("validate","kagglevimgdatafinal").replace("test","kaggletimgdatafinal")
                #--------------------------
                print "place 2"
                #dst_path = "../" + path.rsplit(".",1)[0]+".64x64.jpg"
                #dst_path = dst_path.replace("./train","kaggleimgdatafinal").replace("./validate","kagglevimgdatafinal").replace("./test","kaggletimgdatafinal")
                
                #print "dst_path: ",dst_path
                f = dicom.read_file(path,force=True)
                (PixelSpacingx,PixelSpacingy) = f.PixelSpacing
                (PixelSpacingx,PixelSpacingy) = (float(PixelSpacingx),float(PixelSpacingy))
            
                img = f.pixel_array.astype(float) / np.max(f.pixel_array)
                pixelspacing = PixelSpacingx
                center = (cx,cy)
                #size = 128

                #crop and resize the image
                img = preproc(img, pixelspacing,center)
                print(os.path.dirname(dst_path))
                if not os.path.exists(os.path.dirname(dst_path)):
                    os.makedirs(os.path.dirname(dst_path))
                scipy.misc.imsave(dst_path, img)
                #--------------------Attention remove break----------------------
                #break
            except:
                print(sys.exc_info()[0])
        #---------------Attention remove break--------------------------
        
        counter += 1
        if counter % 100 == 0:
            print("%d slices processed" % counter)
        print("All finished, %d slices in total" % counter)
        return result

#--------------Show dicom image---------------
        #pylab.imshow(f.pixel_array,cmap=pylab.cm.bone)
        #pylab.show()
#-------------------------------------------

#-----------Home-----------------
#root_path = "C:\\Users\\Zheng Zhang\\Desktop\\TestFolder\\1"
#train_csv_path = "C:\\Users\\Zheng Zhang\\Desktop\\TestFolder\\train.csv"
#train_label_csv = "C:\\Users\\Zheng Zhang\\Desktop\\TestFolder\\train-label.csv"

#-----------BEC------------------
root_path = "C:\\Users\\cheung\\Desktop\\TestFolder\\train\\1"
train_csv_path = "C:\\Users\\cheung\\Desktop\\TestFolder\\train.csv"
train_label_csv = "C:\\Users\\cheung\\Desktop\\TestFolder\\train-label.csv"
train_64x64_data = "C:\\Users\\cheung\\Desktop\\TestFolder\\train-64x64-data.csv"

random.seed(10)
frames = get_frames(root_path)
label_map = get_label_map(train_csv_path)

write_label_csv(train_label_csv, frames, get_label_map(train_csv_path))
train_lst = write_data_csv(frames,lambda x,y,z: crop_resize(x,y,z,128))


