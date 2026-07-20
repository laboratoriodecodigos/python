from ultralytics import YOLO
import cv2, numpy as np

VIDEO="C:\\Images\\slow_traffic_small.mp4"
OUT="resultado.mp4"
model=YOLO("yolo11n.pt")
veh={2,3,5,7}
TIMEOUT=30

vehicle_names = {
    2: "Automovil",
    3: "Motocicleta",
    5: "Autobus",
    7: "Camion"
}

def newkf():
    k=cv2.KalmanFilter(4,2)
    k.transitionMatrix=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    k.measurementMatrix=np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    k.processNoiseCov=np.eye(4,dtype=np.float32)*0.01
    k.measurementNoiseCov=np.eye(2,dtype=np.float32)*2
    k.errorCovPost=np.eye(4,dtype=np.float32)*5
    return k

filters={}
last_seen={}
cap=cv2.VideoCapture(VIDEO)
w,h=int(cap.get(3)),int(cap.get(4));fps=cap.get(5) or 30
out=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w+240,h))
frame_no=0
while True:
    ok,frame=cap.read()
    if not ok: break
    frame_no+=1
    ids_activos = {}
    panel=np.full((h,240,3),35,np.uint8)
    res=model.track(frame,persist=True,tracker="bytetrack.yaml",verbose=False)
    if res and res[0].boxes.id is not None:
        b=res[0].boxes
        for box, tid, c in zip(
        b.xyxy.cpu().numpy(),
        b.id.int().cpu().tolist(),
        b.cls.int().cpu().tolist()):

            if c not in vehicle_names:
                continue

            # Guardar ID y tipo de vehículo
            ids_activos[tid] = vehicle_names[c]

            last_seen[tid] = frame_no

            if tid not in filters:
                filters[tid] = newkf()

            x1, y1, x2, y2 = box.astype(int)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            filters[tid].predict()

            filters[tid].correct(
                np.array([
                    [np.float32(cx)],
                    [np.float32(cy)]
                ])
            )

            color = (
                (tid * 53) % 255,
                (tid * 97) % 255,
                (tid * 193) % 255
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"ID {tid}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # Punto rojo en el centro
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    # limpiar ids inactivos
    for tid in list(last_seen):
        if frame_no-last_seen[tid]>TIMEOUT:
            last_seen.pop(tid,None)
            filters.pop(tid,None)
    cv2.putText(panel,"IDS ACTIVOS",(20,35),0,0.8,(255,255,255),2)
    y=70
    for tid in sorted(ids_activos):

        tipo = ids_activos[tid]

        cv2.putText(
            panel,
            f"ID {tid}",
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            panel,
            tipo,
            (90, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        y += 28
    cv2.putText(panel,f"TOTAL {len(ids_activos)}",(20,h-20),0,0.8,(0,255,255),2)
    vis=np.hstack((frame,panel))
    out.write(vis)
    cv2.imshow("Sistema",vis)
    if cv2.waitKey(1)==27: break
cap.release(); out.release(); cv2.destroyAllWindows()
