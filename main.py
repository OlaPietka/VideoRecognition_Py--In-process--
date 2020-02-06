import math   # for mathematical operations
import imageio
import os
import matplotlib.pyplot as plt    # for plotting the images
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from imutils import paths

FRAMES_FOLDER_PATH = "data\\frames\\"
VIDEOS_FOLDER_PATH = "data\\videos\\"

RESIZE = 128
EPOCHS = 2
BS = 1

# Grab the videos paths and randomly shuffle them
video_paths = sorted(list(paths.list_files(VIDEOS_FOLDER_PATH)))

# Initialize the data and labels
data = []
labels = []

for video_path in video_paths:
    # capturing the video from the given path
    reader = imageio.get_reader(video_path)

    FRAME_RATE = math.floor(reader.get_meta_data()["fps"])
    DURATION = reader.get_meta_data()["duration"]

    frames = []
    frames_labels = []
    for frame_id, im in enumerate(reader):
        if frame_id % FRAME_RATE == 0:
            frame = im.copy()
            frame = resize(frame, (RESIZE, RESIZE))
            frame = img_to_array(frame)

            label = video_path.split(os.path.sep)[-2]

            data.append(frame)
            labels.append(label)

    print("Path:", video_path, "| Label:", label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

f = open("model" + ".lbl", "wb")
f.write((pickle.dumps(mlb)))

no_classes = len(mlb.classes_)

(train_data, valid_data, train_labels, valid_labels) = train_test_split(data, labels, test_size=0.25)

print(train_data.shape)
print(valid_data.shape)
print(train_labels.shape)
print(valid_labels.shape)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2,
                         horizontal_flip=True)

# Initialize the model
model = Sequential()
model.add(layers.Embedding(1000, 64))
model.add(layers.LSTM(128))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

# Train the network
H = model.fit(aug.flow(train_data, train_labels, batch_size=BS), validation_data=(valid_data, valid_labels), epochs=EPOCHS, verbose=1)

# Save model to disk
model.save("model" + ".h5")

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("model" + ".png")

print("Done!")
