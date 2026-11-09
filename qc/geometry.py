import cv2
import numpy as np

def scale_coords(width,height,settings):
    sx=width/settings.reference_width; sy=height/settings.reference_height
    poly=np.asarray(settings.polygon_ref,dtype=np.float32)
    poly[:,0]*=sx; poly[:,1]*=sy
    poly=poly.astype(np.int32)
    x1,y1,x2,y2=settings.stamp_roi_ref
    roi=(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy))
    p1=(int(settings.count_line_ref[0][0]*sx),int(settings.count_line_ref[0][1]*sy))
    p2=(int(settings.count_line_ref[1][0]*sx),int(settings.count_line_ref[1][1]*sy))
    return poly,roi,p1,p2

def point_in_polygon(point,polygon):
    return cv2.pointPolygonTest(polygon,(float(point[0]),float(point[1])),False)>=0

def ccw(a,b,c):
    return (c[1]-a[1])*(b[0]-a[0])>(b[1]-a[1])*(c[0]-a[0])

def segments_intersect(a,b,c,d):
    return ccw(a,c,d)!=ccw(b,c,d) and ccw(a,b,c)!=ccw(a,b,d)

def signed_line_side(point,a,b):
    return (b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
