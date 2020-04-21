# FaceDetector.
# Brandon Joffe
# 2016
#
# Copyright 2016, Brandon Joffe, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import dlib
from PIL import Image
import threading
import logging
#import ImageUtils
import time
import numpy as np

if dlib.cuda.get_num_devices()>0:
    print("FaceDetector DLIB using CUDA")
    dlib.DLIB_USE_CUDA = True

modelFile = "models/opencv_face_detector_uint8.pb"
configFile = "models/opencv_face_detector.pbtxt"        

accurate_modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
accurate_configFile = "models/deploy.prototxt"

class FaceDetector(object):
    """This class implements both OpenCV's Haar Cascade
    detector and Dlib's HOG based face detector"""

    def __init__(self):
        self.facecascade = cv2.CascadeClassifier("models/haarcascade_frontalface_alt2.xml")
        self.facecascade2 = cv2.CascadeClassifier("models/haarcascade_frontalface_alt2.xml")
        self.detector = dlib.get_frontal_face_detector()
        self.net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.acc_net = cv2.dnn.readNetFromCaffe(accurate_configFile, accurate_modelFile)
        self.acc_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.acc_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        self.cascade_lock = threading.Lock()
        self.accurate_cascade_lock = threading.Lock()


    def detect_faces(self, image, dlibDetector):
         if dlibDetector:
            return self.detect_dlib_face(image)
         else:
            #return self.detect_cascade_face(image)
            return self.detect_dnn_face(image)

    def pre_processing(self,image):
         """Performs CLAHE on a greyscale image"""
         gray = image
         print("COLOR_BGR2GRAY")
         #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
         cl1 = clahe.apply(gray)
         # cv2.imwrite('clahe_2.jpg',cl1)
         return cl1

    def detect_dnn_face(self, image, accurate=False):
        start = time.time()
        frameHeight, frameWidth, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        if accurate:
            self.acc_net.setInput(blob)
            detections = self.acc_net.forward()
        else:
            self.net.setInput(blob)
            detections = self.net.forward()
        bboxes = []
        confidence = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.9:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                r = dlib.rectangle(x1,y1,x2,y2)
                bboxes.append(r)
        #if bboxes:
        #    print("cv2 dnn ", time.time() - start, bboxes)
        return bboxes
        
    
    def rgb_pre_processing(self,image):
        """Performs CLAHE on each RGB components and rebuilds final
        normalised RGB image - side note: improved face detection not recognition"""
        (h, w) = image.shape[:2]    
        #zeros = np.zeros((h, w), dtype="uint8") # check numpy import
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
        (B, G, R) = cv2.split(image)
        R = clahe.apply(R)
        G = clahe.apply(G)
        B = clahe.apply(B)
     
        filtered = cv2.merge([B, G, R])
        cv2.imwrite('notfilteredRGB.jpg',image)
        cv2.imwrite('filteredRGB.jpg',filtered)
        return filtered


    def detect_dlib_face(self,image):
        # rgbFrame = rgb_pre_processing(rgbFrame)
        image = self.pre_processing(image)
        bbs = self.detector(image, 1)
        # bbs = []
        # dets, scores, idx = self.detector.run(image, 1, -1)
        # for i, d in enumerate(dets):
        #     print("Detection {}, score: {}, face_type:{}".format(
        #         d, scores[i], idx[i]))
        #     if -1*scores[i] < 0.4 or scores[i] > 0:
        #         bbs.append(d)
        #         print "appended: " + str(scores[i])
        #     else:
        #         print "notappended: " + str(scores[i])

        return bbs  

    def detect_cascade_face(self,image):
        #print(">detect_cascadeface")
        with self.cascade_lock:  # Used to block simultaneous access to resource, stops segmentation fault when using more than one camera
            #image = self.pre_processing(image)
            #rects = self.facecascade.detectMultiScale(image, scaleFactor=1.25, minNeighbors=3, minSize=(20, 20), flags = cv2.CASCADE_SCALE_IMAGE)
            rects = self.detect_dnn_face(image, False)
        return rects

    def detect_cascadeface_accurate(self,image):
        """Used to help mitigate false positive detections"""
        print(">detect_cascadeface_accurate")
        with self.accurate_cascade_lock:
            #rects = self.facecascade2.detectMultiScale(img, scaleFactor=1.02, minNeighbors=12, minSize=(20, 20), flags = cv2.CASCADE_SCALE_IMAGE)
            rects = self.detect_dnn_face(image, True)
        return rects
