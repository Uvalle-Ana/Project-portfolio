import cv2
import mediapipe as mp
import numpy as np
import os
import time
import winsound
from tensorflow.keras.models import load_model


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

BASE_DIR = os.path.dirname(__file__)

model = load_model(
    os.path.join(BASE_DIR, "models", "drowsiness_model.h5")
)

classes = ["despierta", "dormida", "bostezo"]

last_alarm_time = 0
drowsy_start_time = None
sleep_counter = 0

def main():
    global last_alarm_time
    global drowsy_start_time
    global sleep_counter

    while True:

        success, frame = camera.read()

        if not success:
            print("No se pudo acceder a la cámara")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        detected_faces = results.multi_face_landmarks

        detected_state = "sin deteccion"
        confidence = 0

        if detected_faces:

            face_points = detected_faces[0]

            frame_height, frame_width = frame.shape[:2]

            x_values = [p.x for p in face_points.landmark]
            y_values = [p.y for p in face_points.landmark]

            x_min = int(min(x_values) * frame_width)
            x_max = int(max(x_values) * frame_width)
            y_min = int(min(y_values) * frame_height)
            y_max = int(max(y_values) * frame_height)

            padding = 20

            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)

            x_max = min(frame_width, x_max + padding)
            y_max = min(frame_height, y_max + padding)

            face_crop = frame[y_min:y_max, x_min:x_max]

            if face_crop.size > 0:

                face_crop = cv2.resize(face_crop, (128, 128))

                face_crop = cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB
                )

                face_crop = face_crop / 255.0

                face_crop = np.expand_dims(
                    face_crop,
                    axis=0
                )

                prediction = model.predict(
                    face_crop,
                    verbose=0
                )

                predicted_index = np.argmax(prediction)

                confidence = np.max(prediction) * 100

                detected_state = classes[predicted_index]

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (180, 120, 255),
                3
            )

        dashboard = np.full(
            (720, 1280, 3),
            (245, 240, 255),
            dtype=np.uint8
        )

        cv2.putText(
            dashboard,
            "Sistema Inteligente de Deteccion de Somnolencia para Conductores",
            (80, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 70, 130),
            2
        )

        camera_view = cv2.resize(frame, (800, 600))

        dashboard[80:680, 20:820] = camera_view

        cv2.rectangle(
            dashboard,
            (850, 80),
            (1250, 680),
            (255, 235, 245),
            -1
        )

        if detected_state == "despierta":
            state_color = (170, 255, 200)
            state_text = "NORMAL"
            risk_text = "BAJO"

        elif detected_state == "bostezo":
            state_color = (255, 240, 180)
            state_text = "FATIGA"
            risk_text = "MEDIO"

        elif detected_state == "dormida":
            state_color = (255, 180, 200)
            state_text = "SOMNOLENCIA"
            risk_text = "ALTO"

        else:
            state_color = (220, 220, 220)
            state_text = "---"
            risk_text = "---"

        cv2.putText(
            dashboard,
            "ESTADO ACTUAL",
            (930, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (100, 70, 130),
            2
        )

        cv2.circle(
            dashboard,
            (930, 230),
            30,
            state_color,
            -1
        )

        cv2.putText(
            dashboard,
            state_text,
            (980, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 70, 130),
            2
        )

        cv2.putText(
            dashboard,
            f"Confianza: {confidence:.1f}%",
            (900, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 70, 130),
            2
        )

        cv2.putText(
            dashboard,
            f"Riesgo: {risk_text}",
            (900, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (100, 70, 130),
            2
        )

        if detected_state == "bostezo":

            cv2.rectangle(
                dashboard,
                (880, 520),
                (1220, 610),
                (255, 240, 180),
                -1
            )

            cv2.putText(
                dashboard,
                "FATIGA DETECTADA",
                (905, 575),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (120, 90, 150),
                3
            )

        # Lógica de alarma con tiempo (igual hay que ajustar elapsed porque aun no me convence)
        if detected_state == "dormida" and confidence > 85:
            sleep_counter += 1
            current_time = time.time()
            if drowsy_start_time is None:
                drowsy_start_time = current_time
            elapsed = current_time - drowsy_start_time

            if elapsed >= 1.5:
                cv2.putText(
                    dashboard,
                    "ALARMA ACTIVADA",
                    (900, 460),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                if current_time - last_alarm_time > 3:
                    winsound.Beep(2500, 1000)
                    last_alarm_time = current_time
        else:
            sleep_counter = 0
            drowsy_start_time = None

        cv2.imshow(
            "Sistema Inteligente de Deteccion de Somnolencia",
            dashboard
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()