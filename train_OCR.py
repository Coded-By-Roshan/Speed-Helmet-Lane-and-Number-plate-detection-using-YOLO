import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

dataset_path = 'final_dataset'

def load_data(dataset_path):
    images, labels = [], []
    label_dict = {}
    label_index = 0

    for folder_name in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            label_dict[label_index] = folder_name  
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (32, 32))  
                    images.append(img)
                    labels.append(label_index)
            label_index += 1

    return np.array(images), np.array(labels), label_dict

X, y, label_dict = load_data(dataset_path)
X = X.astype('float32') / 255.0
y = tf.keras.utils.to_categorical(y, num_classes=len(label_dict))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_dict), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                    height_shift_range=0.1, zoom_range=0.1)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)

steps_per_epoch = math.ceil(len(X_train) / 32)
validation_steps = math.ceil(len(X_val) / 32)

model.fit(train_generator,
          validation_data=val_generator,
          steps_per_epoch=steps_per_epoch,
          validation_steps=validation_steps,
          epochs=10)

if not os.path.exists("models"):
    os.makedirs("models")

model.save("models/NEW_OCR_MODEL.keras")

val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_val, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_dict.values(), yticklabels=label_dict.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

report = classification_report(y_true, y_pred, target_names=label_dict.values())
print("Classification Report:")
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)
