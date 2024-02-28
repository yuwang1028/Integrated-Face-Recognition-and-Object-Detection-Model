"""-----------------------------------------------------------
 2. collect other face data sets
There are Yale face library of Yale University, ORL face library of Cambridge University, FERET face library of the United States Department of Defense and so on
This system USES face data set download: http://vis-www.cs.umass.edu/lfw/lfw.tgz
First put the downloaded photo set in img_source directory, and use dlib to batch identify the face part of the image.
And save to the specified directory faces_other
Face size: 64*64
----------------------------------------------------------------"""
# -*- codeing: utf-8 -*-
import sys
import cv2
import os
import dlib

source_path = './img_source'
faces_other_path = './faces_other'
size = 64
if not os.path.exists(faces_other_path):
    os.makedirs(faces_other_path)

"""frontal_face_detector"""
detector = dlib.get_frontal_face_detector()

num = 1
"""其中./path/dirnames/filenames"""
for (path, dirnames, filenames) in os.walk(source_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % num)
            img_path = path+'/'+filename
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            """dets will be the return """
            dets = detector(gray_img, 1)

            """--------------------------------------------------------------------
            The enumerate function is used to traverse the elements in the sequence and their subscripts. i is the face number and d is the element corresponding to i. 
            left: The distance between the left side of the face and the left edge of the picture; right: The distance between the right side of the face and the left edge of the image 
            top: the distance between the upper part of the face and the upper part of the image; bottom: The distance between the bottom of the face and the top border of the picture
            ----------------------------------------------------------------------"""
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1:y1,x2:y2]
                face = cv2.resize(face, (size,size))   # adjust the size
                cv2.imshow('image',face)
                cv2.imwrite(faces_other_path+'/'+str(num)+'.jpg', face)
                num += 1

            key = cv2.waitKey(30)
            if key == 27:
                sys.exit(0)