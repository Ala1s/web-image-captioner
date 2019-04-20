from utilites import get_captions, read_pickle, generate_batch, encode_captions
import json
import os
from google.colab import drive
import tensorflow as tf
from tensorflow.contrib import keras

mount_directory = "/content/gdrive"
drive.mount(mount_directory,force_remount=True)

train_img_embeddings = read_pickle("/content/gdrive/My Drive/ICC 2019/train_img_embeddings.pickle")
train_img_fns = read_pickle("/content/gdrive/My Drive/ICC 2019/train_img_fns.pickle") #filenames
val_img_embeddings = read_pickle("/content/gdrive/My Drive/ICC 2019/val_img_embeddings.pickle")
val_img_fns = read_pickle("/content/gdrive/My Drive/ICC 2019/val_img_fns.pickle") #filenames

train_captions = get_captions(train_img_fns, "/content/gdrive/My Drive/ICC 2019/captions_train-val2014.zip", 
                                      "annotations/captions_train2014.json")

val_captions = get_captions(val_img_fns, "/content/gdrive/My Drive/ICC 2019/captions_train-val2014.zip", 
                                      "annotations/captions_val2014.json")
#load dictionaries
with open('/content/gdrive/My Drive/ICC 2019/word2idx8K.json') as fp:
  word2idx = json.load(fp)
with open('/content/gdrive/My Drive/ICC 2019/idx2word8K.json') as fp:
  idx2word = json.load(fp)
final_model = keras.models.load_model('/content/gdrive/My Drive/withglovev2.h5')

train_captions_indexed = encode_captions(train_captions, word2idx)
val_captions_indexed = encode_captions(val_captions, word2idx)

train_generator = generate_batch(train_img_embeddings,train_captions_indexed,8)
valid_generator = generate_batch(val_img_embeddings,val_captions_indexed,8)

final_model.fit_generator(train_generator, steps_per_epoch=10250, epochs = 12, verbose = 1, validation_data=valid_generator, validation_steps=5000)
final_model.save('/content/gdrive/My Drive/withglovev2trained.h5')