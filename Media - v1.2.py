"""
Brant Jiang

#Collection of media formats for analysis#
1) General Media
2) Pictures
3) Videos

#LOGS#
1/1/22 - Est doc and TODO
1/7/22 - Est Photo framework
1/8/22 - Gray scale stuff
1/13/22 - Begin cracking thresholding

#TODO#
-Thresholding
-Video

#MISC#
Env -- LookingGlass
Check main doc in drive for more info
Opencv works in BGR colorspace
.png cause opencv to shit itself (channel discrepancy due to alpha?)
Differences in transparency properties can cause concatenating images to mess up 
Solution is to compress as jpg 
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

class Photo():
    #PARAMS: filename, scale factor, image object
    def __init__(self, *args):
        self._name=args[0]
        self._scaleFactor=args[1]
        
        if len(args)<=2:
            #cv2.IMREAD_COLOR -> 1 //Ignores transparency
            #cv2.IMREAD_GRAYSCALE -> 0 //Grayscale
            #cv2.IMREAD_UNCHANGED -> -1 //Includes transparency (alpha channel)
            self._image=cv2.imread(args[0],-1)
        elif isinstance(args[2],np.ndarray) and not(args[2] is None):
            self._image=args[2]
            self._nHeight=int(self._image.shape[0]*self._scaleFactor)
            self._nWidth=int(self._image.shape[1]*self._scaleFactor)
            
            #Converted back to color for channel equity with BGR images
            self._gray=cv2.cvtColor(self._image,cv2.COLOR_BGR2GRAY)
            #Slight blur to remove noise (image, kernel size for averaging, std deviation perimtted [if 0, calculated from kernel size])
            #Thresholds must be positive and odd, smaller values for higher res is probably better
            self._thresh=cv2.GaussianBlur(self._gray, (101,101), 0)
            
            self._collection=[self._image,self._gray,self._thresh]
                
        
    def dispImg(self, *args):
        if len(args)>0:
            i=args[0]
        else:
            i=0
        
        self._showImg(self._name,self._collection[i])
        
        
    def _showImg(self, name, img):
        #Creates auto fitted window (to make multiple windows just use diff win name)
        #cv2.imshow("Name of window", img)
        #cv2.WINDOW_AUTOSIZE is default for namedWindow();
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, (self._nWidth, self._nHeight))
        cv2.imshow(name, img)
        
        #waits n milliseconds for keyboard event to continue program, 0 -> waits indef
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        
    def dispHist(self, *args):
        if len(args)>0:
            i=args[0]
        else:
            i=0
            
        if i<1:
            self._showHist(self._name, self._collection[i],True)
        else:
            if i<2:
                name="Gray "+self._name
            else:
                name=self._name+" Thresholded"
            self._showHist(name, self._collection[i],False)  
        
        
    def _showHist(self, name, img, flag):
        window=plt.figure()
        window.canvas.set_window_title(name)
        plt.title(label="Histogram of "+name)

        if flag:
            color=("blue","green","red")
            for i, col in enumerate(color):
                hist=cv2.calcHist([img],[i],None,[256],[0,256])
                plt.plot(hist, color=col)
                plt.xlim([0,256])
        else:
            hist=cv2.calcHist([cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)],[0],None,[256],[0,256])
            plt.plot(hist, color="gray")
            plt.xlim([0,256])
            
        plt.show()
        
        
    def compImg(self, i, j):
        img1=self._collection[i]
        img2=self._collection[j]
        
        #Ensure channel sameness in case a pic is grayscale
        if(i>0):
            img1=cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        if(j>0):
            img2=cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
            
        toShow=Photo("Comparison of "+self._name,self._scaleFactor/2,np.concatenate((img1,img2),axis=1))
        toShow.dispImg()
        
        self.dispHist(i)
        self.dispHist(j)
        
    
    def compThresh(self, other, motionThresh=4):
        if other is None:
            return False
        
        delta=cv2.absdiff(self._thresh, other._thresh)
        #self._showImg("Delta", delta)
        threshDelta=cv2.threshold(delta,25,255,cv2.THRESH_BINARY)[1]
        #self._showImg("Thresh", threshDelta)
        threshDelta=cv2.dilate(threshDelta, None, iterations=2)
        #self._showImg("Dilate", threshDelta)
        
        delta=cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(threshDelta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            (x,y,w,h)=cv2.boundingRect(contour)
            
            if(cv2.contourArea(contour) > motionThresh): 
                cv2.rectangle(self._collection[0],(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(delta,(x,y),(x+w,y+h),(0,255,0),2)
            
        toShow=Photo("Motion Detector",self._scaleFactor/2,np.concatenate((self._collection[0],delta),axis=1))
        toShow.dispImg()

    def checkMotion(self, other, motionThresh=8):
        if other is None:
            return False
        
        threshDelta=cv2.threshold(cv2.absdiff(self._thresh, other._thresh),25,255,cv2.THRESH_BINARY)[1]
        threshDelta=cv2.dilate(threshDelta, None, iterations=2)
        contours, _ = cv2.findContours(threshDelta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            (x,y,w,h)=cv2.boundingRect(contour)
            
            if(cv2.contourArea(contour) > motionThresh):     
                return True
            
        return False


class Video():
    #PARAMS: filename, scale factor, video object
    def __init__(self, *args):
        self._name=args[0]
        self._scaleFactor=args[1]
        
        if len(args)<=2:
            self._video=cv2.VideoCapture(args[0])
        elif isinstance(args[2],cv2.VideoCapture):
            self._video=args[2]
        
        self._nHeight=int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT)*self._scaleFactor)
        self._nWidth=int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH)*self._scaleFactor)
        self._fps=self._video.get(cv2.CAP_PROP_FPS)
        self._frames=int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))
        self._len=self._frames/self._fps
        
    
    def play(self):
        _, frame=self._video.read()
        
        while not(frame is None):
            Photo(self._name,self._scaleFactor,frame).dispImg()
            cv2.waitKey(int(1.0/self._fps*1000))
            _, frame=self._video.read()
    
    
    def _getTime(self, fNum):
        return fNum/self._fps
    
    
    def findMotion(self):
        f1=None
        f2=None
        fNum=0
        motionPeriods=[]
        start=-1
        
        for i in range (self._frames): 
            fNum+=1
            #print(self._getTime(fNum))
            f2=f1
            _, frame=self._video.read()
            f1=Photo(self._name, self._scaleFactor, frame)
            f1.compThresh(f2)
            cv2.waitKey(int(1.0/self._fps*1000))
            
            if(f1.checkMotion(f2,200)):
                if start==-1:
                    #print("MOTION STARTED")
                    start=self._getTime(fNum+1)
                #print("MOTION")
            else:
                if start!=-1:
                    #print("MOTION ENDED")
                    motionPeriods.append((start,self._getTime(fNum-1)))
                    start=-1
                #print("NO MOTION")
                    
        if(start!=-1):
            motionPeriods.append((start,self._getTime(self._frames)))
        
        return motionPeriods
            
                    
    def __str__(self):
        toRet="Length: "+str(self._len)+" sec \nFPS: "+str(self._fps)+"\n"+str(int(self._nWidth/self._scaleFactor))+"x"+str(int(self._nHeight/self._scaleFactor))
        return toRet

    
