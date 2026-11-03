"""Course roadmap figure of AI Technologies (Basic Course), AIT-BC.

Draws assets/course_roadmap.png: 5 topics, 19 lessons, 40 hours, notebooks NB01-NB09.
Run from the repository root:  python build/make_roadmap.py
Copyright 2026 Oleksii Fesenko. Licensed under the Apache License, Version 2.0.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
S1,S2,S3=12,9,7.5
TOPIC_COL=["#0072B2","#009E73","#E69F00","#CC79A7","#D55E00"]
def light(c,a):
    r,g,b=mpl.colors.to_rgb(c); return (1-a+a*r,1-a+a*g,1-a+a*b)
def star(ax,cx,cy,ro,ri,**kw):
    th=np.linspace(np.pi/2,np.pi/2+2*np.pi,11)[:-1]; rr=np.where(np.arange(10)%2==0,ro,ri)
    ax.add_patch(Polygon(np.c_[cx+rr*np.cos(th),cy+rr*np.sin(th)],closed=True,**kw))
topics=[
 dict(n=1,title="Computer vision\nin Google Colab",h=8,nb="NB01–NB02",
      lessons=[("1.1","L","Tasks of CV for UAV and UGV","NB01"),("1.2","G","Google Colab tools for imagery","NB01"),
               ("1.3","P","First work with EO frames","NB02"),("1.4","P","Video streams of UAVs and UGVs","NB02")],
      data="HIT-UAV · Kust4K · M3OT\nVisDrone · GOOSE", res="Course project on Google Drive;\nframes and video of UAVs"),
 dict(n=2,title="Digital image and\npreprocessing",h=8,nb="NB03–NB04",
      lessons=[("2.1","L","Digital image and video stream","NB03"),("2.2","G","Stages of frame preprocessing","NB03"),
               ("2.3","P","Channels, ROI and letterbox","NB04"),("2.4","P","Filtering: noise, blur, CLAHE","NB04")],
      data="HIT-UAV · Kust4K · M3OT\nVisDrone · GOOSE", res="Frames prepared for a neural\nnetwork: channels, ROI, filters"),
 dict(n=3,title="Classical feature\nextraction",h=6,nb="NB05",
      lessons=[("3.1","L","Edges, silhouettes, contours","NB05"),("3.3","P","Thermal signatures","NB05"),
               ("3.4","P","Stabilisation of FPV video","NB05")],
      data="HIT-UAV · Kust4K\nM3OT · TII-RATM (FPV)", res="Warm-object detector;\nstabilised FPV video"),
 dict(n=4,title="Machine learning and\nneural networks",h=8,nb="NB06–NB07",
      lessons=[("4.1","L","ML recognition of military objects","NB06"),("4.2","G","From neuron to MobileNet, ONNX","NB06"),
               ("4.3","P","HOG + SVM thermal verifier","NB07"),("4.4","P","MobileNet transfer learning","NB07")],
      data="Military vehicles (24 types)\nSULAND mines · HIT-UAV", res="Thermal target verifier;\nvehicle recogniser in ONNX"),
 dict(n=5,title="Detection, segmentation\nand tracking",h=10,nb="NB08–NB09",
      lessons=[("5.1","L","YOLO: IoU, mAP, small targets","NB08"),("5.2","G","UGV traversable area, ByteTrack","NB08"),
               ("5.3","P","YOLO11 on UAV, FPV, thermal","NB09"),("5.4","M","Counter-UAV detector + tracker","NB09")],
      data="DUT Anti-UAV · M3OT · GOOSE\nTII-RATM · HIT-UAV", res="Trained drone detector\nand tracker (YOLO11 + ByteTrack)"),
]
assert sum(t["h"] for t in topics)==40 and sum(len(t["lessons"]) for t in topics)==19
assert sum(t["h"] for t in topics[:3])==22 and sum(t["h"] for t in topics[3:])==18
assert all(t["h"]==sum(4 if l[1]=="M" else 2 for l in t["lessons"]) for t in topics)
W,Hgt=16,9.4
fig=plt.figure(figsize=(W,Hgt)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,Hgt); ax.set_aspect("equal"); ax.axis("off")
xs=[2.0,5.0,8.0,11.0,14.0]; ROAD_Y=6.15; CW=2.82
ax.text(0.35,Hgt-0.42,"Course roadmap",fontsize=S1+4,fontweight="bold",color="#1f2a37",va="center")
ax.text(0.35,Hgt-0.85,"AI Technologies (Basic Course) · 5 topics · 19 lessons · 40 hours · 9 Google Colab notebooks on real UAV and UGV data",
        fontsize=S2+1,color="#4b5563",va="center")
for x0,x1,lab in ((xs[0]-CW/2,xs[2]+CW/2,"Module 1 · Topics 1–3 · 22 h"),(xs[3]-CW/2,xs[4]+CW/2,"Module 2 · Topics 4–5 · 18 h")):
    ax.add_patch(FancyBboxPatch((x0,7.72),x1-x0,0.42,boxstyle="round,pad=0,rounding_size=0.21",fc="#eef2f7",ec="none"))
    ax.text((x0+x1)/2,7.93,lab,ha="center",va="center",fontsize=S2,color="#374151",fontweight="bold")
yst=lambda x: ROAD_Y+0.13*np.sin((x-0.35)/3.0*np.pi)
xr=np.linspace(0.45,15.62,600)
ax.plot(xr,yst(xr),color="#9aa5b1",lw=22,solid_capstyle="round",zorder=1)
ax.plot(xr,yst(xr),color="white",lw=1.6,ls=(0,(6,6)),zorder=2)
ax.add_patch(Circle((0.45,yst(0.45)),0.2,fc="#1f2a37",ec="white",lw=2,zorder=3))
ax.text(0.45,yst(0.45)-0.5,"Start",ha="center",va="center",fontsize=S3,color="#1f2a37",fontweight="bold")
star(ax,15.62,yst(15.62),0.34,0.15,fc="#f5b700",ec="#1f2a37",lw=1.2,zorder=3)
ax.text(15.62,yst(15.62)-0.55,"Final\nmini-project",ha="center",va="center",fontsize=S3,color="#1f2a37",fontweight="bold")
TYPE={"L":"Lecture","G":"Group class","P":"Practical","M":"Mini-project"}
for t,x,c in zip(topics,xs,TOPIC_COL):
    y0=yst(x)
    ax.add_patch(Circle((x,y0),0.42,fc=c,ec="white",lw=3,zorder=4))
    ax.text(x,y0,str(t["n"]),ha="center",va="center",fontsize=S1+3,color="white",fontweight="bold",zorder=5)
    ax.text(x,7.05,t["title"],ha="center",va="center",fontsize=S1,fontweight="bold",color=c,linespacing=1.15)
    ax.text(x+0.55,y0+0.36,f"{t['h']} h",ha="left",va="center",fontsize=S2,color="#1f2a37",fontweight="bold")
    ax.text(x+0.55,y0-0.34,t["nb"],ha="left",va="center",fontsize=S3,color="#4b5563")
    top=5.35
    for num,typ,title,nb in t["lessons"]:
        ch=0.78 if typ=="M" else 0.66; yb=top-ch
        fc=light(c,0.10) if typ!="M" else light("#f5b700",0.28)
        ax.add_patch(FancyBboxPatch((x-CW/2,yb),CW,ch,boxstyle="round,pad=0,rounding_size=0.1",fc=fc,ec=light(c,0.55),lw=1.0,zorder=3))
        cx,cy=x-CW/2+0.27,yb+ch/2; tc=None
        if typ=="L":   ax.add_patch(Circle((cx,cy),0.17,fc=c,ec=c,lw=1.2,zorder=4)); tc="white"
        elif typ=="G": ax.add_patch(Circle((cx,cy),0.17,fc=light(c,0.45),ec=c,lw=1.2,zorder=4)); tc="#1f2a37"
        elif typ=="P": ax.add_patch(Circle((cx,cy),0.17,fc="white",ec=c,lw=1.6,zorder=4)); tc=c
        else: star(ax,cx,cy,0.21,0.09,fc="#f5b700",ec="#1f2a37",lw=0.8,zorder=4)
        if tc: ax.text(cx,cy,typ,ha="center",va="center",fontsize=S3,fontweight="bold",color=tc,zorder=5)
        ax.text(x-CW/2+0.53,yb+ch-0.2,f"{num}  {TYPE[typ]} · {'4 h' if typ=='M' else '2 h'}",ha="left",va="center",fontsize=S3,color="#4b5563",zorder=5)
        ax.text(x-CW/2+0.53,yb+0.21,title,ha="left",va="center",fontsize=S2,color="#1f2a37",zorder=5)
        ax.add_patch(FancyBboxPatch((x+CW/2-0.62,yb+ch-0.33),0.52,0.22,boxstyle="round,pad=0,rounding_size=0.08",fc=c,ec="none",zorder=4))
        ax.text(x+CW/2-0.36,yb+ch-0.22,nb,ha="center",va="center",fontsize=S3-1,color="white",fontweight="bold",zorder=5)
        top=yb-0.1
    ax.text(x-CW/2,1.98,t["data"],ha="left",va="center",fontsize=S3,color="#374151",linespacing=1.3)
    ax.text(x-CW/2,1.1,t["res"],ha="left",va="center",fontsize=S3,color=c,fontweight="bold",linespacing=1.3)
ax.text(0.35,1.98,"Data",ha="left",va="center",fontsize=S3,color="#9aa5b1",fontweight="bold",rotation=90)
ax.text(0.35,1.1,"Result",ha="left",va="center",fontsize=S3,color="#9aa5b1",fontweight="bold",rotation=90)
ax.plot([0.6,15.4],[1.55,1.55],color="#e5e7eb",lw=1)
lx,ly=0.6,0.42; c0="#4b5563"
for i,(typ,lab) in enumerate([("L","Lecture, 2 h"),("G","Group class, 2 h"),("P","Practical class, 2 h"),("M","Mini-project, 4 h")]):
    cx=lx+0.17+i*2.35
    if typ=="L": ax.add_patch(Circle((cx,ly),0.15,fc=c0,ec=c0)); ax.text(cx,ly,"L",ha="center",va="center",fontsize=S3-1,color="white",fontweight="bold")
    elif typ=="G": ax.add_patch(Circle((cx,ly),0.15,fc=light(c0,0.45),ec=c0)); ax.text(cx,ly,"G",ha="center",va="center",fontsize=S3-1,color="#1f2a37",fontweight="bold")
    elif typ=="P": ax.add_patch(Circle((cx,ly),0.15,fc="white",ec=c0,lw=1.4)); ax.text(cx,ly,"P",ha="center",va="center",fontsize=S3-1,color=c0,fontweight="bold")
    else: star(ax,cx,ly,0.19,0.08,fc="#f5b700",ec="#1f2a37",lw=0.8)
    ax.text(cx+0.3,ly,lab,ha="left",va="center",fontsize=S3,color="#374151")
ax.text(15.4,ly,"+ practical notebook: all five topics in one Colab notebook",ha="right",va="center",fontsize=S3,color="#374151")

if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)
    fig.savefig("assets/course_roadmap.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    print("saved assets/course_roadmap.png")
