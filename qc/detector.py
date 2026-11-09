import cv2
import numpy as np
from .geometry import point_in_polygon

# HSV ranges are deliberately configurable: lighting and camera white balance vary.
DEFAULT_RANGES={
    "red":[[0,65,45,12,255,255],[168,65,45,180,255,255]],
    "gold":[[12,65,65,38,255,255]],
    "silver":[[0,0,90,180,85,255]]
}
BGR_COLORS={"red":(60,40,255),"gold":(0,210,255),"silver":(220,225,225),"unknown":(200,120,255)}

class ItemDetector:
    def __init__(self,settings):
        self.settings=settings
        self.ranges=settings.class_thresholds or DEFAULT_RANGES

    def _mask_for(self,hsv,label):
        mask=np.zeros(hsv.shape[:2],np.uint8)
        for values in self.ranges.get(label,[]):
            lo=np.array(values[:3],np.uint8); hi=np.array(values[3:],np.uint8)
            mask=cv2.bitwise_or(mask,cv2.inRange(hsv,lo,hi))
        return mask

    def detect(self,frame,polygon,stamp_roi):
        h,w=frame.shape[:2]
        zone=np.zeros((h,w),np.uint8)
        cv2.fillPoly(zone,[polygon],255)
        x1,y1,x2,y2=stamp_roi
        roi_mask=np.zeros_like(zone)
        x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
        if x2>x1 and y2>y1: roi_mask[y1:y2,x1:x2]=255
        zone=cv2.bitwise_and(zone,roi_mask)
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        detections=[]
        for label in ("red","gold","silver"):
            mask=cv2.bitwise_and(self._mask_for(hsv,label),zone)
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area=cv2.contourArea(contour)
                if not self.settings.min_area<=area<=self.settings.max_area: continue
                perimeter=cv2.arcLength(contour,True)
                circularity=(4*np.pi*area/(perimeter*perimeter)) if perimeter>0 else 0
                if circularity<self.settings.min_circularity: continue
                (cx,cy),radius=cv2.minEnclosingCircle(contour)
                center=(int(cx),int(cy))
                if not point_in_polygon(center,polygon): continue
                x,y,bw,bh=cv2.boundingRect(contour)
                detections.append({"center":center,"radius":max(4,int(radius)),
                    "bbox":(x,y,bw,bh),"color":label,"area":float(area),
                    "circularity":float(circularity)})
        # Merge duplicate detections caused by overlapping HSV classes.
        unique=[]
        for d in sorted(detections,key=lambda x:x["area"],reverse=True):
            if all(np.linalg.norm(np.array(d["center"])-np.array(u["center"]))>12 for u in unique):
                unique.append(d)
        return unique
