import os
from google.colab import drive
import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
import tqdm
from utilites import save_pickle, img_from_raw, crop_image, preprocess_input
import cv2

mount_directory = "/content/gdrive"
drive.mount(mount_directory,force_remount=True)

model = keras.applications.InceptionV3(include_top=False)
model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
train2014 = zipfile.ZipFile("train2014.zip")
train_img_fns = []
train_img_embeddings = []
#takes a lot of time
for file in tqdm(train2014.namelist()):
	if os.path.splitext(file)[-1] in (".jpg",):
		raw_bytes=train2014.read(file)
		img = img_from_raw(raw_bytes)
		img = crop_image(img)
    	img = cv2.resize(img, (299, 299))
    	img = img.astype("float32")
    	img = preprocess_input(img)
        train_img_fns.append(file)
        train_img_embeddings.append(model.predict(img))

save_pickle(train_img_embeddings, "train_img_embeddings.pickle")
save_pickle(train_img_fns, "train_img_fns.pickle")

val2014 = zipfile.ZipFile("val2014.zip")
val_img_fns = []
val_img_embeddings = []
#takes a lot of time
for file in tqdm(val2014.namelist()):
	if os.path.splitext(file)[-1] in (".jpg",):
		img = img_from_raw(raw_bytes)
		img = crop_image(img)
    	img = cv2.resize(img, (299, 299))
    	img = img.astype("float32")
    	img = preprocess_input(img)
        val_img_fns.append(file)
        val_img_embeddings.append(model.predict(img))

save_pickle(val_img_embeddings, "val_img_embeddings.pickle")
save_pickle(val_img_fns, "val_img_fns.pickle")




