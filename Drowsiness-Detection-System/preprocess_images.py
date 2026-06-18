import cv2
import mediapipe as mp
import os


INPUT_FOLDER = "img"
OUTPUT_FOLDER = "dataset_procesado"

IMAGE_SIZE = (128, 128)


mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

classes = ["despierta", "dormida", "bostezo"]

for class_name in classes:

    output_path = os.path.join(
        OUTPUT_FOLDER,
        class_name
    )

    os.makedirs(output_path, exist_ok=True)

processed = 0
failed = 0

for class_name in classes:

    input_class_folder = os.path.join(
        INPUT_FOLDER,
        class_name
    )

    output_class_folder = os.path.join(
        OUTPUT_FOLDER,
        class_name
    )

    files = os.listdir(input_class_folder)

    for file_name in files:

        image_path = os.path.join(
            input_class_folder,
            file_name
        )

        image = cv2.imread(image_path)

        if image is None:
            failed += 1
            continue

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = face_detector.process(rgb_image)

        if not results.detections:
            failed += 1
            continue

        detection = results.detections[0]

        bbox = detection.location_data.relative_bounding_box

        h, w, _ = image.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)

        width = int(bbox.width * w)
        height = int(bbox.height * h)

        margin = 20

        x = max(0, x - margin)
        y = max(0, y - margin)

        width = min(
            w - x,
            width + 2 * margin
        )

        height = min(
            h - y,
            height + 2 * margin
        )

        face = image[
            y:y+height,
            x:x+width
        ]

        if face.size == 0:
            failed += 1
            continue

        face = cv2.resize(
            face,
            IMAGE_SIZE
        )

        output_file = os.path.join(
            output_class_folder,
            os.path.splitext(file_name)[0] + ".jpg"
        )

        cv2.imwrite(
            output_file,
            face
        )

        processed += 1

print("\nProceso terminado")
print(f"Imágenes procesadas: {processed}")
print(f"Imágenes descartadas: {failed}")