import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# CONFIGURACIÓN

VIDEO_PATH = 'C:/Images/Trafico.mp4'

MODEL_PATH = 'yolov8n.pt'

LINEA_CONTEO = 500

OFFSET = 10

# Clases COCO
# car=2 motorcycle=3 bus=5 truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

# CARGAR MODELO

model = YOLO(MODEL_PATH)

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.3
)

# ABRIR VIDEO

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

delay = int(1000 / fps)

contador = 0

vehiculos_contados = set()

# Guarda posición anterior
trayectorias = {}

# LOOP PRINCIPAL

while True:

    inicio_frame = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    # DETECCIÓN YOLO
   
    results = model(frame, verbose=False)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            if cls not in VEHICLE_CLASSES:
                continue

            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            detections.append(
                (
                    [x1, y1, w, h],
                    conf,
                    cls
                )
            )

    # TRACKING
   
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    # LÍNEA DE CONTEO
  
    cv2.line(
        frame,
        (100, LINEA_CONTEO),
        (1200, LINEA_CONTEO),
        (0, 255, 255),
        3
    )

    # RECORRER TRACKS

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # GUARDAR POSICIÓN ANTERIOR

        if track_id not in trayectorias:

            trayectorias[track_id] = cy

        prev_y = trayectorias[track_id]

        # DIRECCIÓN

        moving_down = cy > prev_y

        # Actualizar posición
        trayectorias[track_id] = cy

        # CONTAR SOLO SI BAJA

        if moving_down:

            if (
                LINEA_CONTEO - OFFSET
                < cy
                < LINEA_CONTEO + OFFSET
            ):

                if track_id not in vehiculos_contados:

                    vehiculos_contados.add(track_id)

                    contador += 1

                    print(
                        f'Vehículo contado: ID {track_id}'
                    )

                    # Línea verde al contar
                    cv2.line(
                        frame,
                        (100, LINEA_CONTEO),
                        (1200, LINEA_CONTEO),
                        (0, 255, 0),
                        5
                    )

        # DIBUJAR BOX

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Centro
        cv2.circle(
            frame,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

        # ID
        cv2.putText(
            frame,
            f'ID {track_id}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    # FPS

    fin_frame = time.time()

    fps_real = 1 / (fin_frame - inicio_frame)

    # TEXTO

    cv2.putText(
        frame,
        f'Conteo: {contador}',
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 255),
        3
    )

    cv2.imshow("Conteo Un Sentido", frame)

    key = cv2.waitKey(delay)

    if key == 27:
        break


# FINALIZAR

cap.release()

cv2.destroyAllWindows()

