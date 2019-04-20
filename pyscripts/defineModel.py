import os
from google.colab import drive
import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
import json

mount_directory = "/content/gdrive"
drive.mount(mount_directory,force_remount=True)
L = keras.layers

IMG_EMBED_SIZE = 2048
BOTTLENECK = 256
WORD_EMBED_SIZE = 200
DROPOUT_RATE = 0.5
start_indx = 2
end_indx = 0
pad_idx = 1
MAX_LEN = 20

#load dictionaries
with open('/content/gdrive/My Drive/ICC 2019/word2idx8K.json') as fp:
  word2idx = json.load(fp)
with open('/content/gdrive/My Drive/ICC 2019/idx2word8K.json') as fp:
  idx2word = json.load(fp)

embeddings_index = {} # empty dictionary
f = open('/content/gdrive/My Drive/glove.6B.200d.txt', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(vocab), WORD_EMBED_SIZE))
for word, i in word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Not found words will have zero vectors
        embedding_matrix[i] = embedding_vector

image_input = L.Input(shape=(IMG_EMBED_SIZE,),name='image_embedding_input')
image_dropout = L.Dropout(DROPOUT_RATE, name = 'image_dropout')
image_bottleneck = L.Dense(BOTTLENECK, activation = 'relu', name = 'image_bottleneck')
image_model = image_bottleneck(image_dropout(image_input))

caption_input = L.Input(shape=(MAX_LEN,),name='caption_input')
caption_embedding = L.Embedding(len(vocab), WORD_EMBED_SIZE, mask_zero=True, name='embedding_with_glove')
caption_dropout = L.Dropout(DROPOUT_RATE, name='caption_dropout')
caption_lstm = L.LSTM(BOTTLENECK, name='caption_lstm')
caption_model = caption_lstm(caption_dropout(caption_embedding(caption_input)))

decoder_input = L.add([image_model,caption_model])
decoder_dense = L.Dense(BOTTLENECK, activation='relu', name='decoder_dense')
decoder_output = L.Dense(len(vocab),activation='softmax', name='output')
output = decoder_output(decoder_dense(decoder_input))
final_model = keras.models.Model(inputs=[image_input,caption_input],outputs=output)
final_model.layers[2].set_weights([embedding_matrix])
final_model.layers[2].trainable = False
final_model.compile(loss='categorical_crossentropy',  optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['categorical_accuracy'])
final_model.save('/content/gdrive/My Drive/withglovev2.h5')