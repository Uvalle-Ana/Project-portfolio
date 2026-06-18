import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical

dataset_path = "dataset_procesado"

classes = [
    "despierta",
    "dormida",
    "bostezo"
]

image_size = 128

images = []
labels = []

for class_index, class_name in enumerate(classes):

    class_path = os.path.join(
        dataset_path,
        class_name
    )

    print(f"Cargando {class_name}...")

    for image_name in os.listdir(class_path):

        image_path = os.path.join(
            class_path,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(
            image,
            (image_size, image_size)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        images.append(image)
        labels.append(class_index)

print("\nCarga terminada")
print("Imágenes cargadas:", len(images))
print("Etiquetas cargadas:", len(labels))

X = np.array(images) / 255.0
y = np.array(labels)

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDatos de entrenamiento:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    )
)

model.add(
    MaxPooling2D((2, 2))
)

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)

model.add(
    MaxPooling2D((2, 2))
)

model.add(
    Flatten()
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        3,
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    validation_data=(X_test, y_test)
)

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/drowsiness_model.h5"
)

print("\nModelo guardado correctamente")