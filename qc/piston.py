import cv2
import numpy as np

class PistonMonitor:
    """Motion proxy for the pneumatic stamp/piston ROI; calibrate threshold on site."""
    def __init__(self,threshold=8.0):
        self.threshold=threshold; self.previous=None; self.state="IDLE"; self.score=0.0
    def update(self,frame,roi):
        x1,y1,x2,y2=roi
        crop=frame[max(0,y1):max(1,y2),max(0,x1):max(1,x2)]
        if crop.size==0: return self.state,self.score
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(5,5),0)
        if self.previous is not None and self.previous.shape==gray.shape:
            diff=cv2.absdiff(gray,self.previous)
            self.score=float(np.mean(diff))
            self.state="ACTIVE" if self.score>=self.threshold else "IDLE"
        self.previous=gray
        return self.state,self.score
