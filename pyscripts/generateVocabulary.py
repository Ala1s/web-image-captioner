import json
from utilites import read_pickle, get_captions, split_cap, 

train_img_fns = read_pickle("/content/gdrive/My Drive/ICC 2019/train_img_fns.pickle")
val_img_fns = read_pickle("/content/gdrive/My Drive/ICC 2019/val_img_fns.pickle")
    
train_captions = get_captions(train_img_fns, "/content/gdrive/My Drive/ICC 2019/captions_train-val2014.zip", 
                                      "annotations/captions_train2014.json")

val_captions = get_captions(val_img_fns, "/content/gdrive/My Drive/ICC 2019/captions_train-val2014.zip", 
                                      "annotations/captions_val2014.json")

#truncate captions longer than 20 words
for captions_id in range(len(train_captions)):
  for caption_id in range(len(train_captions[captions_id])):
    if len(split_cap(train_captions[captions_id][caption_id]))>20:
      train_captions[captions_id][caption_id] = ' '.join(split_cap(train_captions[captions_id][caption_id])[:20])

word2idx = generate_vocabulary(train_captions)
idx2word = {idx: word for word, idx in word2idx.items()}
with open('/content/gdrive/My Drive/ICC 2019/word2idx8K.json', 'w') as fp:
    json.dump(vocab, fp)
with open('/content/gdrive/My Drive/ICC 2019/idx2word8K.json', 'w') as fp:
    json.dump(vocab_inverse, fp)

    