#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
from PIL import Image
import os
import string
import tensorflow
from pickle import dump
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras.applications.xception import preprocess_input
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.text import Tokenizer #for text tokenization
from tensorflow.keras.utils import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense#Keras to build our CNN and LSTM
from keras.layers import LSTM, Embedding, Dropout
from tqdm.notebook import tqdm #to check loop progress
tqdm().pandas()


# In[33]:


tensorflow.__version__


# In[34]:


conda list cudnn


# In[35]:


pwd


# In[36]:


#!unzip Flickr8k_Dataset.zip
#!unzip Flickr8k_text.zip


# In[37]:


# Loading a text file into memory
def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# get all imgs with their captions
def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions ={}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [ caption ]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

#Data cleaning- lower casing, removing puntuations and words containing numbers
def cleaning_text(captions):
    table = str.maketrans('','',string.punctuation)
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):

            img_caption.replace("-"," ")
            desc = img_caption.split()

            #converts to lowercase
            desc = [word.lower() for word in desc]
            #remove punctuation from each token
            desc = [word.translate(table) for word in desc]
            #remove hanging 's and a 
            desc = [word for word in desc if(len(word)>1)]
            #remove tokens with numbers in them
            desc = [word for word in desc if(word.isalpha())]
            #convert back to string

            img_caption = ' '.join(desc)
            captions[img][i]= img_caption
    return captions

def text_vocabulary(descriptions):
    # build vocabulary of all unique words
    vocab = set()

    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]

    return vocab

#All descriptions in one file 
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc )
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()


# Set these path according to project folder in you system
dataset_text = ""
dataset_images = "Flicker8k_Dataset"

#we prepare our text data
filename = "Flickr8k.token.txt"
#loading the file that contains all data
#mapping them into descriptions dictionary img to 5 captions
descriptions = all_img_captions(filename)
print("Length of descriptions =" ,len(descriptions))

#cleaning the descriptions
clean_descriptions = cleaning_text(descriptions)

#building vocabulary 
vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))

#saving each description to file 
#save_descriptions(clean_descriptions, "descriptions.txt")


# In[ ]:





# In[38]:


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def extract_features(directory):
        model = Xception( include_top=False, pooling='avg' )
        features = {}
        for img in tqdm(os.listdir(directory)):
            filename = directory + "/" + img
            image = Image.open(filename)
            image = image.resize((299,299))
            image = np.expand_dims(image, axis=0)
            #image = preprocess_input(image)
            image = image/127.5
            image = image - 1.0

            feature = model.predict(image)
            features[img] = feature
        return features

#2048 feature vector
#features = extract_features(dataset_images)
#dump(features, open("features.p","wb"))


# In[39]:


features=load(open("features.p","rb"))


# In[40]:


#load the data 
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos


def load_clean_descriptions(filename, photos): 
    #loading clean_descriptions
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):

        words = line.split()
        if len(words)<1 :
            continue

        image, image_caption = words[0], words[1:]

        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)

    return descriptions


def load_features(photos):
    #loading all features
    all_features = load(open("features.p","rb"))
    #selecting only needed features
    features = {k:all_features[k] for k in photos}
    return features


filename ="Flickr_8k.trainImages.txt"

#train = loading_data(filename)
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)


# In[ ]:





# In[41]:


#converting dictionary to clean list of descriptions
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

#creating tokenizer class 
#this will vectorise text corpus
#each integer will represent token in dictionary

from keras.preprocessing.text import Tokenizer

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
#dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
vocab_size


# In[42]:


#calculate maximum length of descriptions
def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)
    
max_length = max_length(descriptions)
max_length


# In[43]:


#create input-output sequence pairs from the image description.

#data generator, used by model.fit_generator()
def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            #retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
            yield [[input_image, input_sequence], output_word]

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

#You can check the shape of the input and output for your model
[a,b],c = next(data_generator(train_descriptions, features, tokenizer, max_length))
a.shape, b.shape, c.shape
#((47, 2048), (47, 32), (47, 7577))


# In[44]:


from keras.layers import Layer
import tensorflow as tf
class BahdanauAttentionLayer(Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def build(self, input_shape):
        super(BahdanauAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        features, hidden = inputs
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.units), (input_shape[0][0], input_shape[0][1], 1)


# In[45]:


from keras.utils import plot_model
def define_model(vocab_size, max_length):
    # Features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, return_sequences=True, return_state=True)(se2)
    lstm_out, _, _ = se3

    # Attention layer
    attention_layer = BahdanauAttentionLayer(256)
    context_vector, _ = attention_layer([lstm_out, fe2])

    # Merging both models
    decoder1 = add([fe2, context_vector])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)

    return model


# In[ ]:


# train our model
print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)

model = define_model(vocab_size, max_length)
epochs = 10
steps = len(train_descriptions)
# making a directory models to save our models
#os.mkdir("models_atten")
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit_generator(generator, epochs=1, steps_per_epoch= steps, verbose=1)
    model.save("models_atten/model_" + str(i) + ".h5")


# In[ ]:





# In[46]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse


def extract_features(filename, model):
  try:
    image = Image.open(filename)
  except:
    print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
  image = image.resize((299,299))
  image = np.array(image)
  # for images that has 4 channels, we convert them into 3 channels
  if image.shape[2] == 4: 
      image = image[..., :3]
  image = np.expand_dims(image, axis=0)
  image = image/127.5
  image = image - 1.0
  feature = model.predict(image)
  return feature

def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None


def generate_desc(model, tokenizer, photo, max_length):
  in_text = 'start'
  for i in range(max_length):
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_length)
    pred = model.predict([photo,sequence], verbose=0)
    pred = np.argmax(pred)
    word = word_for_id(pred, tokenizer)
    if word is None:
        break
    in_text += ' ' + word
    if word == 'end':
        in_text += ' '
        break
  return in_text


# In[30]:


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    attention_weights_list = []
    
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred, attention_weights = model.predict([photo, sequence], verbose=0)
        attention_weights_list.append(attention_weights[0])
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            in_text += ' '
            break
    
    return in_text, np.array(attention_weights_list)


# In[24]:


def plot_attention(image_path, words, attention_weights):
    image = Image.open(image_path)
    image = image.resize((299, 299))
    img_array = np.array(image)

    fig = plt.figure(figsize=(10, 10))
    len_words = len(words)

    for i in range(len_words):
        weights = np.reshape(attention_weights[i], (8, 8))
        weights = np.expand_dims(weights, axis=-1)
        weights = np.tile(weights, (1, 1, 3))
        weights = Image.fromarray((weights * 255).astype(np.uint8))
        weights = weights.resize((299, 299), resample=Image.BICUBIC)

        img_att = Image.blend(image, weights, alpha=0.5)
        ax = fig.add_subplot(len_words//2, len_words//2, i+1)
        ax.set_title(words[i], fontsize=12)
        plt.imshow(img_att)
        plt.axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()


# In[25]:


model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})


# In[26]:


tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")


# In[27]:


image_path = 'Flicker8k_Dataset/3385593926_d3e9c21170.jpg'
photo = extract_features(image_path,xception_model)  # Use the pre-trained CNN model for feature extraction
generated_desc, attention_weights = generate_desc(model, tokenizer, photo, max_length)

# Remove 'start' and 'end' tokens from the generated description
words = generated_desc.split()
words = words[1:-1]

# Plot the attention weights on the input image
plot_attention(image_path, words, attention_weights)


# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:


from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu

scorer = rouge_scorer.RougeScorer(['rougeL'])
ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)

rouge_scores = {"precision": [], "recall": [], "f1measure": []}
all_references = []
all_candidates = []

for img in test_descriptions:
    references = [desc.split() for desc in ref_descriptions[img]]
    candidate = test_descriptions[img].split()

    # Calculate Rouge scores
    best_precision, best_recall, best_f1measure = 0, 0, 0
    for desc in ref_descriptions[img]:
        rouge_score = scorer.score(test_descriptions[img], desc)
        precision, recall, f1measure = rouge_score["rougeL"]
        
        if f1measure > best_f1measure:
            best_precision, best_recall, best_f1measure = precision, recall, f1measure

    rouge_scores["precision"].append(best_precision)
    rouge_scores["recall"].append(best_recall)
    rouge_scores["f1measure"].append(best_f1measure)

    # Prepare lists for BLEU score calculation
    all_references.append(references)
    all_candidates.append(candidate)

# Calculate BLEU scores
bleu_score = corpus_bleu(all_references, all_candidates)

# Calculate average Rouge scores
avg_rouge_precision = sum(rouge_scores["precision"]) / len(rouge_scores["precision"])
avg_rouge_recall = sum(rouge_scores["recall"]) / len(rouge_scores["recall"])
avg_rouge_f1measure = sum(rouge_scores["f1measure"]) / len(rouge_scores["f1measure"])

print("BLEU Score:", bleu_score)
print("Average Rouge Precision:", avg_rouge_precision)
print("Average Rouge Recall:", avg_rouge_recall)
print("Average Rouge F1-measure:", avg_rouge_f1measure)


# In[ ]:





# In[ ]:





# In[43]:


photo


# In[49]:


test_images = load_photos("Flickr_8k.testImages.txt")


# In[19]:


test_images = load_photos("Flickr_8k.testImages.txt")
# img_path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'

test_descriptions = {}
for img in test_images:
  img_path = "Flicker8k_Dataset/"+img
  max_length = 32

  photo = extract_features(img_path, xception_model)
  # img = Image.open(img_path)

  description = generate_desc(model, tokenizer, photo, max_length)
  test_descriptions[img] = description


# In[54]:


from keras.layers.pooling.average_pooling2d import AvgPool2D
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu

scorer = rouge_scorer.RougeScorer(['rougeL'])
ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)

avg_rouge_scores = {"precision":[], "recall":[], "f1measure":[]}
avg_bleu_scores = []

for img in test_descriptions:
  desc_rouge_scores = {"precision":[], "recall":[], "f1measure":[]}
  # desc_bleu_scores = []
  avg_bleu_score = sentence_bleu([i.split() for i in ref_descriptions[img]], test_descriptions[img].split())
  for desc in ref_descriptions[img]:
    rouge_score = scorer.score(test_descriptions[img], desc)
    desc_rouge_scores["precision"].append(rouge_score["rougeL"][0])
    desc_rouge_scores["recall"].append(rouge_score["rougeL"][1])
    desc_rouge_scores["f1measure"].append(rouge_score["rougeL"][2])
    # desc_bleu_scores.append(bleu_score)  
  
  # avg_bleu_score = sum(desc_bleu_scores)/len(desc_bleu_scores)
  avg_rouge_precision = sum(desc_rouge_scores["precision"])/len(desc_rouge_scores["precision"])
  avg_rouge_recall = sum(desc_rouge_scores["recall"])/len(desc_rouge_scores["recall"])
  avg_rouge_f1measure = sum(desc_rouge_scores["f1measure"])/len(desc_rouge_scores["f1measure"])

  avg_bleu_scores.append(avg_bleu_score)
  avg_rouge_scores["precision"].append(avg_rouge_precision)
  avg_rouge_scores["recall"].append(avg_rouge_recall)  
  avg_rouge_scores["f1measure"].append(avg_rouge_f1measure)    


# In[ ]:


#################


# In[ ]:


for img in test_descriptions:
  i.split() for i in ref_descriptions[img]]
   test_descriptions[img].split()]


# In[ ]:





# In[52]:


import json
#with open("models_atten/predictions.json","w") as f:
#    json.dump(test_descriptions, f)


# In[ ]:





# In[53]:


with open("models_atten/predictions.json","r") as f:
    test_descriptions=json.load(f)


# In[22]:


print("BLEU score: ", sum(avg_bleu_scores)/len(avg_bleu_scores))
print("ROUGE precision: ", sum(avg_rouge_scores["precision"])/len(avg_rouge_scores["precision"]))
print("ROUGE recall: ", sum(avg_rouge_scores["recall"])/len(avg_rouge_scores["recall"]))
print("ROUGE f1: ", sum(avg_rouge_scores["f1measure"])/len(avg_rouge_scores["f1measure"]))


# In[ ]:





# In[21]:


img_path = 'Flicker8k_Dataset/3385593926_d3e9c21170.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(f"       Model with Attention:{description}")
plt.imshow(img)
plt.show()


# In[25]:


img_path = 'Flicker8k_Dataset/466956209_2ffcea3941.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(f"       Model with Attention:{description}")
plt.imshow(img)
plt.show()


# In[ ]:





# In[18]:


img_path = 'Flicker8k_Dataset/316833109_6500b526dc.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# In[56]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['3385593926_d3e9c21170.jpg']
test_caption=[test_descriptions['3385593926_d3e9c21170.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


img_path = 'Flicker8k_Dataset/3385593926_d3e9c21170.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(f"Actual:{reference_caption}")
print(f"Predicted:{test_caption}")
print(f"Bleu_Score:{score}")
plt.imshow(img)
plt.show()


# In[120]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['2105756457_a100d8434e.jpg']
test_caption=[test_descriptions['2105756457_a100d8434e.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


img_path = 'Flicker8k_Dataset/2105756457_a100d8434e.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(f"Reference:{reference_caption}")
print(f"Predicted:{description}")
print(f"Bleu_Score:{score}")
plt.imshow(img)
plt.show()


# In[121]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['2105756457_a100d8434e.jpg']
test_caption=[test_descriptions['2105756457_a100d8434e.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


img_path = 'Flicker8k_Dataset/2105756457_a100d8434e.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(f"Reference:{reference_caption}")
print(f"Predicted:{description}")
print(f"Bleu_Score:{score}")
plt.imshow(img)
plt.show()


# In[ ]:


count = 0
for img_name in test_descriptions:
  if count>5:
    break
  count+=1
  img_path = "Flicker8k_Dataset/"+img_name
  img = Image.open(img_path)
  print("\n\n")
  plt.imshow(img)
  print(img_name)
  print("Actual: ", ref_descriptions[img_name])
  print("Prediction:, ", test_descriptions[img_name])
  plt.show()


# In[100]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['3154641421_d1b9b8c24c.jpg']
test_caption=[test_descriptions['3154641421_d1b9b8c24c.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])
img_path = 'Flicker8k_Dataset/3154641421_d1b9b8c24c.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
print(f"Bleu_Score:{score}")
plt.imshow(img)


# In[110]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['3320356356_1497e53f80.jpg']
test_caption=[test_descriptions['3320356356_1497e53f80.jpg']]
corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


# In[111]:


ref_descriptions['3320356356_1497e53f80.jpg']


# In[112]:


test_caption


# In[ ]:





# In[82]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['3453259666_9ecaa8bb4b.jpg']
test_caption=[test_descriptions['3453259666_9ecaa8bb4b.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


img_path = 'Flicker8k_Dataset/3453259666_9ecaa8bb4b.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
print(f"Bleu_Score:{score}")
plt.imshow(img)


# In[83]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['968081289_cdba83ce2e.jpg']
test_caption=[test_descriptions['968081289_cdba83ce2e.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


img_path = 'Flicker8k_Dataset/968081289_cdba83ce2e.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
print(f"Bleu_Score:{score}")
plt.imshow(img)


# In[ ]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['968081289_cdba83ce2e.jpg']
test_caption=[test_descriptions['968081289_cdba83ce2e.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


img_path = 'Flicker8k_Dataset/968081289_cdba83ce2e.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
print(f"Bleu_Score:{score}")
plt.imshow(img)


# In[84]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['2280525192_81911f2b00.jpg']
test_caption=[test_descriptions['2280525192_81911f2b00.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


img_path = 'Flicker8k_Dataset/2280525192_81911f2b00.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
print(f"Bleu_Score:{score}")
plt.imshow(img)


# In[85]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['2683963310_20dcd5e566.jpg']
test_caption=[test_descriptions['2683963310_20dcd5e566.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


img_path = 'Flicker8k_Dataset/2683963310_20dcd5e566.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
print(f"Bleu_Score:{score}")
plt.imshow(img)


# In[86]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['525863257_053333e612.jpg']
test_caption=[test_descriptions['525863257_053333e612.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


img_path = 'Flicker8k_Dataset/525863257_053333e612.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
print(f"Bleu_Score:{score}")
plt.imshow(img)


# In[94]:


ref_descriptions = load_clean_descriptions("descriptions.txt", test_images)
reference_caption=ref_descriptions['311146855_0b65fdb169.jpg']
test_caption=[test_descriptions['311146855_0b65fdb169.jpg']]
score=corpus_bleu([[ref.split() for ref in reference_caption]], [cand.split() for cand in test_caption])


img_path = 'Flicker8k_Dataset/311146855_0b65fdb169.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
print(f"Bleu_Score:{score}")
plt.imshow(img)


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


img_path = 'Flicker8k_Dataset/3637013_c675de7705.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_temp/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# In[21]:


img_path = 'Flicker8k_Dataset/23445819_3a458716c1.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# In[22]:


img_path = 'Flicker8k_Dataset/42637987_866635edf6.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# In[23]:


img_path = 'Flicker8k_Dataset/95728664_06c43b90f1.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# In[24]:


img_path = 'Flicker8k_Dataset/109202756_b97fcdc62c.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# In[25]:


img_path = 'Flicker8k_Dataset/109260216_85b0be5378.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# In[26]:


img_path = 'Flicker8k_Dataset/3517124784_4b4eb62a7a.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# In[27]:


img_path = 'Flicker8k_Dataset/316833109_6500b526dc.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_atten/model_9.h5',custom_objects={'BahdanauAttentionLayer': BahdanauAttentionLayer})
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# In[ ]:




