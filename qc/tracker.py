import math
from collections import deque
from .geometry import signed_line_side

class CentroidTracker:
    def __init__(self,max_distance=65,max_missed=12):
        self.max_distance=max_distance; self.max_missed=max_missed
        self.next_id=1; self.tracks={}

    def update(self,detections):
        assignments={}; used=set()
        candidates=[]
        for tid,tr in self.tracks.items():
            for di,d in enumerate(detections):
                dist=math.dist(tr["center"],d["center"])
                if dist<=self.max_distance: candidates.append((dist,tid,di))
        for _,tid,di in sorted(candidates):
            if tid not in assignments and di not in used:
                assignments[tid]=di; used.add(di)
        for tid,di in assignments.items():
            tr=self.tracks[tid]; d=detections[di]
            tr["previous"]=tr["center"]; tr["center"]=d["center"]
            tr["color"]=d["color"]; tr["radius"]=d["radius"]
            tr["bbox"]=d["bbox"]; tr["missed"]=0
        for tid in list(self.tracks):
            if tid not in assignments:
                self.tracks[tid]["missed"]+=1
                if self.tracks[tid]["missed"]>self.max_missed: del self.tracks[tid]
        for di,d in enumerate(detections):
            if di in used: continue
            self.tracks[self.next_id]={"id":self.next_id,"center":d["center"],
                "previous":None,"color":d["color"],"radius":d["radius"],
                "bbox":d["bbox"],"missed":0,"counted":False,"history":deque(maxlen=8)}
            self.next_id+=1
        for tr in self.tracks.values():
            tr["history"].append(tr["center"])
        return list(self.tracks.values())

    def crossed(self,track,line_a,line_b):
        prev=track.get("previous")
        if prev is None: return False
        before=signed_line_side(prev,line_a,line_b)
        after=signed_line_side(track["center"],line_a,line_b)
        return before==0 or after==0 or (before<0<after) or (after<0<before)
