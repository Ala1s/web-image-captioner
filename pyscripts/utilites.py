import cv2
import zipfile
import json
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
from collections import defaultdict
from collections import Counter # dict that allows us to count number of occurences of strings (tokens)
from itertools import chain # returns one item from first iterator, then from second and so on until iterators are exhausted
import re


PAD = "#PAD#" # for sentence padding
UNK = "#UNK#" # Unknown word, out of vocabulary
START = "#START#" # Marker for start of sentence
END = "#END#" # Marker for end of sentence

def read_pickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)

def save_pickle(outs, file_name):
    with open(file_name, "wb") as f:
	pickle.dump(outs, f, protocol=pickle.HIGHEST_PROTOCOL)
	
def get_captions(file_names, zip_file_name, zip_json_path):
    zf = zipfile.ZipFile(zip_file_name)
    j = json.loads(zf.read(zip_json_path).decode("utf8"))
    id_to_file_name = {img["id"]: img["file_name"] for img in j["images"]}
    file_name_to_captions = defaultdict(list)
    for cap in j['annotations']:
        file_name_to_captions[id_to_file_name[cap['image_id']]].append(cap['caption'])
    file_name_to_captions = dict(file_name_to_caps)
    return list(map(lambda x: file_name_to_captions[x], file_names))

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def crop_image(img):
    h, w = img.shape[0], img.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2
    return img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]

def img_from_raw(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def generate_batch(input_images, output_text, num_photos_in_batch):
  image_embeds,partial_caps,next_words = list(), list(), list()
  n = 0
  while True:
    for image_id in range(input_images.shape[0]):
      n+=1
      image = input_images[image_id]
      captions = output_text[image_id]
      for cap in captions:
        for i in range(1, len(cap)):
          in_seq, out_seq = cap[:i], cap[i]
          in_seq = pad_sequences([in_seq], maxlen=20, padding='post', value = 1.0)[0]
          out_seq = to_categorical([out_seq], num_classes=len(vocab))[0]
          image_embeds.append(image)
          partial_caps.append(in_seq)
          next_words.append(out_seq)
      if n == num_photos_in_batch:
        yield ([np.asarray(image_embeds), np.asarray(partial_caps)], np.asarray(next_words))
        image_embeds, partial_caps, next_words = list(), list(), list()
        n=0

def split_cap(caption):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', caption.lower())))

def generate_vocabulary(captions):
    c = Counter(chain(*map(split_cap, chain(*train_captions))))
    word2idx = set()
    for i in c:
      if c[i] >= 5:
        word2idx.add(i)
    word2idx.update([PAD, UNK, START, END])
    return {token: index for index, token in enumerate(sorted(vocab))}
    
def encode_captions(captions, word2idx):
    res = list()
    for captions_i in captions:
      chunk = list()
      for sentence in captions_i:
        a = split_cap(sentence)
        ind = [word2idx[START], *(word2idx[symb] if symb in word2idx else word2idx[UNK] for symb in a), word2idx[END]]
        chunk.append(ind)
      res.append(chunk)
    return res


