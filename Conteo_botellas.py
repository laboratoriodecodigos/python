
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import time

VIDEO_PATH="C:\\Images\\botellas2.mp4"
OUTPUT_VIDEO="resultado_botellas.mp4"
OUTPUT_XLSX="reporte_botellas.xlsx"

model=YOLO("yolo11n.pt")
BOTTLE_CLASS=39

cap=cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit("No se pudo abrir el video")

w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=cap.get(cv2.CAP_PROP_FPS) or 30

LINE_X=int(w*0.5)

fourcc=cv2.VideoWriter_fourcc(*"mp4v")
out=cv2.VideoWriter(OUTPUT_VIDEO,fourcc,fps,(w+300,h))

count=0
counted=set()
log=[]
start=time.time()

last_x = {}
counted = set()

while True:
    ok,frame=cap.read()
    if not ok:
        break

    results=model.track(frame,persist=True,tracker="botsort.yaml",conf=0.35,verbose=False)

    cv2.line(frame,(LINE_X,0),(LINE_X,h),(0,255,255),2)

    if results and results[0].boxes.id is not None:
        boxes=results[0].boxes.xyxy.cpu().numpy()
        ids=results[0].boxes.id.cpu().numpy().astype(int)
        cls=results[0].boxes.cls.cpu().numpy().astype(int)

        for box,tid,c in zip(boxes,ids,cls):
            if c!=BOTTLE_CLASS:
                continue
            x1,y1,x2,y2=map(int,box)
            if (x2-x1)<10 or (y2-y1)<20:
                continue
            cx=(x1+x2)//2
            cy=(y1+y2)//2
            color = (
                    np.random.default_rng(tid).integers(0,255),
                    np.random.default_rng(tid+1).integers(0,255),
                    np.random.default_rng(tid+2).integers(0,255)
                )
            color = tuple(map(int, color))
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"ID {tid}",(x1,y1-5),0,0.5,color,2)
            cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
            if tid in last_x:

                # Cruce de DERECHA -> IZQUIERDA
                if last_x[tid] > LINE_X and cx <= LINE_X:

                    if tid not in counted:
                        counted.add(tid)
                        count += 1
                        log.append({
                            "id": tid,
                            "time_s": round(time.time()-start, 2),
                            "total": count
                        })

            # Guardar la posición actual para el siguiente frame
            last_x[tid] = cx

    panel=np.zeros((h,300,3),dtype=np.uint8)
    bpm=count/((time.time()-start)/60) if time.time()>start else 0
    texts=[("BOTTLE COUNTER",40), (f"Total: {count}",100),
           (f"BPM: {bpm:.1f}",150),
           (f"IDs: {len(counted)}",200),
           (f"Tiempo: {int(time.time()-start)} s",250)]
    for t,y in texts:
        cv2.putText(panel,t,(15,y),0,0.8,(0,255,0),2)
    outframe=np.hstack((frame,panel))
    out.write(outframe)
    cv2.imshow("Conteo",outframe)
    if cv2.waitKey(1)==27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
pd.DataFrame(log).to_excel(OUTPUT_XLSX,index=False)
print("Total:",count)
