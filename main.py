import argparse, time
from pathlib import Path
import cv2
from qc.config import load_settings
from qc.geometry import scale_coords
from qc.detector import ItemDetector
from qc.tracker import CentroidTracker
from qc.piston import PistonMonitor
from qc.logger import EventLogger
from qc.visualizer import draw_overlay

def main():
    ap=argparse.ArgumentParser(description="Conveyor line color quality inspection")
    ap.add_argument("--source",default="0",help="Camera index or video file")
    ap.add_argument("--config",default="config/settings.json")
    ap.add_argument("--output",default="",help="Annotated MP4 output")
    ap.add_argument("--no-display",action="store_true")
    args=ap.parse_args()
    cfg=load_settings(args.config)
    source=int(args.source) if args.source.isdigit() else args.source
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f"Cannot open video source: {args.source}")
    ok,first=cap.read()
    if not ok: raise SystemExit("No frames received from source.")
    polygon,roi,line_a,line_b=scale_coords(first.shape[1],first.shape[0],cfg)
    detector=ItemDetector(cfg); tracker=CentroidTracker(cfg.max_match_distance,cfg.track_max_missed)
    piston=PistonMonitor(cfg.piston_motion_threshold)
    out_path=args.output or (str(Path(cfg.output_dir)/"quality_control.mp4") if cfg.save_video else "")
    writer=None
    if out_path:
        Path(out_path).parent.mkdir(parents=True,exist_ok=True)
        h,w=first.shape[:2]
        writer=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS) or 25,(w,h))
    logger=EventLogger(Path(cfg.output_dir)/"events.csv")
    counts={"red":0,"gold":0,"silver":0,"unknown":0}
    frame=first; last=time.time(); fps=0.0
    print("Quality-control monitor started. Press Q/Esc to exit.")
    try:
        while True:
            now=time.time(); dt=now-last; last=now
            if dt>0: fps=.9*fps+.1*(1/dt) if fps else 1/dt
            detections=detector.detect(frame,polygon,roi)
            tracks=tracker.update(detections)
            for tr in tracks:
                if tr["missed"]==0 and not tr.get("counted") and tracker.crossed(tr,line_a,line_b):
                    tr["counted"]=True
                    color=tr["color"] if tr["color"] in counts else "unknown"
                    counts[color]+=1
                    logger.log(tr["id"],color,tr["center"],"COUNT_LINE_CROSSED")
            state,score=piston.update(frame,roi)
            view=draw_overlay(frame.copy(),polygon,line_a,line_b,roi,tracks,counts,state,score,fps)
            if writer: writer.write(view)
            if not args.no_display:
                cv2.imshow("Quality Control Conveyor Line Monitor",view)
                if cv2.waitKey(1)&0xFF in (ord("q"),27): break
            ok,frame=cap.read()
            if not ok: break
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
    print("Final counts:",counts)
    print("Event log:",Path(cfg.output_dir)/"events.csv")

if __name__=="__main__": main()
