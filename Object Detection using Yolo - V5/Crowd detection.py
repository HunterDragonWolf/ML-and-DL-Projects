#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# The latest versions of the libraries are being considered for the processes to be followed. The libraries were installed via standard pip method.

# In[1]:


import cv2
import numpy as np
import time


#    

# - CV2 = cv2 is the module import name for opencv-python
# - NumPy = NumPy is a Python library used for working with arrays.
# - time = The Python time module provides many ways of representing time in code, such as objects, numbers, and strings.

# In[2]:


np.random.seed(20)                    ## Same color will be assigned for a class for multiple runs of the code.

## Create Object Detection Class = "class Detector"

class Detector:
    
    ## Create a constructor for the class
    ## We assign the arguments (videoPath, configPath, modelPath, classesPath) to class wise variables.
    ## These variables will be used and will be available in all the methods that will be implemented within this class.
    
    
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        
        ## Initialising the network and set some parameters.
        
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)                       ## Image size on which the model was trained.
        self.net.setInputScale(1.0/127.5)                    ## Images to be scaled between 1 to -1.
        self.net.setInputMean((127.5, 127.5, 127.5))         ## Against each channel the mean value is subtracted.
        self.net.setInputSwapRB(True)                        ## Swap R and B channels to convert images from BGR to RGB.
        
        self.readClasses()                                   ## The Class method being called within init method.
    
    ## Read Class Label List from COCO.NAMES
    ## Define another method
    
    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()          ## Store all the entries in the file as list inside a variable
            
        self.classesList.insert(0, '__Background__')          ## Model predicts the index 0 as background and hence an addition.
        
        ## Define random colors using NumPy, for all 3 channels.
        
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
    
        print(self.classesList)
    
    ## Define another method: onVideo
    
    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)                 ## Way to open a video using cv2.
        
        if (cap.isOpened()==False):
            print("Error opening the file")
            return
                                                               ## The frame is stored in a variable called image.
        (success, image) = cap.read()                          ## If the video doesn't open correctly, we return.
                                                               ## Otherwise we read the first frame from the video.
        startTime = 0
        
        ## As long as the frame is successfully captured:
        
        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime
            
            ## We provide the captured image to the network and define confidence threshold for detection.
            ## bboxs = Bounding boxes found in the image
            
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.6)
            
            ## Converting the bounding boxs and confidence into a list.
            
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))
            
            ## A Non-Maximus-Supression on bounding boxes is implemented, it eliminates all the overlapping bboxs.
            ## nms_threshold returns the indexes of the bboxs that have an overlap below that threshold.
            
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.6)
            
            ## Checking if the length of non overlapping bboxs is not 0, then proceed with drawing of bboxs.
            
            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    
                    ## Extract bounding box from original list.
                    
                    bbox = bboxs[np.squeeze(bboxIdx[i])]                            ## Squeeze to get integer and not array.
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])]) ## Extract class label id against bbox index.
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]     ## Values converted to int and extract color.
                    
                    ## Create Display text with 2 decimals.                          
                                              
                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                    
                    ## Unpack the bbox to get the x and y coordinates with width and height,
                    ## the info is used to draw box on the coordinates.
                    
                    x,y,w,h = bbox
                    
                    ## Pass the staring coordinates of bbox (x,y) and the ending coordinates (x+w,y+h),
                    ## to the rectangle method of the cv2 module.
                    ## Color of the box is set to white, with approp. thickness.
                                              
                    cv2.rectangle(image, (x,y), (x+w, y+h), color = classColor, thickness = 1)
                    
                    ## cv2.putText will help display the text on the top left corner with desired edits.
                                              
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                    
                    ## Beautifying of the bounding boxes.
                                              
                    lineWidth = min(int(w * 0.3), int(h * 0.3))
                                                       
                    cv2.line(image, (x,y), (x + lineWidth, y), classColor, thickness = 4)
                    cv2.line(image, (x,y), (x, y + lineWidth), classColor, thickness = 4)
                    
                    cv2.line(image, (x + w,y), (x + w - lineWidth, y), classColor, thickness = 4)
                    cv2.line(image, (x + w,y), (x + w, y + lineWidth), classColor, thickness = 4)
                    
                    cv2.line(image, (x,y + h), (x + lineWidth, y + h), classColor, thickness = 4)
                    cv2.line(image, (x,y + h), (x, y + h - lineWidth), classColor, thickness = 4)
                    
                    cv2.line(image, (x + w,y + h), (x + w - lineWidth, y + h), classColor, thickness = 4)
                    cv2.line(image, (x + w,y + h), (x + w, y + h - lineWidth), classColor, thickness = 4)
            
            ## To display the fps on the left corner of the screen.
                                              
            cv2.putText(image, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            
            ## Show the frame                                  
                                              
            cv2.imshow("Result", image)
            
            ## Mechanism to break off the loop.
                                              
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            (success, image) = cap.read()
        cv2.destroyAllWindows()             ## Destroying cv2 windows at the end.


# In[3]:


## All the methods in the Detector class are used here.

def main():
    videoPath = "texas.mp4"                                           ## Path of the video
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"       ## Path of the configured model
    modelPath = "frozen_inference_graph.pb"
    classesPath = "coco1.names"                                        ## Path to class labels list
    
    ## Initialise detector class which takes the 4 four smentioned parameters.
    
    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()                                              ## Call on the onVideo method.
    
## Call the main function    
    
if __name__ == '__main__':
    main()


# In[ ]:




