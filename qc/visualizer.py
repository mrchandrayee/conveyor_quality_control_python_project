import cv2
from .detector import BGR_COLORS

def draw_overlay(frame,polygon,line_a,line_b,roi,tracks,counts,piston_state,piston_score,fps):
    cv2.polylines(frame,[polygon],True,(0,255,80),2)
    cv2.line(frame,line_a,line_b,(255,255,0),2)
    x1,y1,x2,y2=roi
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,165,0),1)
    for tr in tracks:
        if tr["missed"]>0: continue
        x,y=tr["center"]; color=BGR_COLORS.get(tr["color"],BGR_COLORS["unknown"])
        cv2.circle(frame,(x,y),tr["radius"]+7,(0,255,255),2)
        cv2.circle(frame,(x,y),tr["radius"],color,2)
        cv2.putText(frame,f"#{tr['id']} {tr['color'].upper()}",(x-35,y-tr["radius"]-12),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    panel=frame.copy()
    cv2.rectangle(panel,(8,8),(230,132),(10,18,24),-1)
    cv2.addWeighted(panel,.72,frame,.28,0,frame)
    cv2.putText(frame,"QUALITY CONTROL",(18,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,100),1)
    cv2.putText(frame,f"TOTAL  {sum(counts.values()):04d}",(18,57),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,180,255),2)
    cv2.putText(frame,f"RED {counts['red']}  GOLD {counts['gold']}",(18,82),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
    cv2.putText(frame,f"SILVER {counts['silver']}  OTHER {counts['unknown']}",(18,103),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
    color=(0,255,128) if piston_state=="ACTIVE" else (0,165,255)
    cv2.putText(frame,f"STAMP: {piston_state} ({piston_score:.1f})",(18,123),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
    cv2.putText(frame,f"FPS {fps:.1f}",(frame.shape[1]-100,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,0),1)
    return frame
