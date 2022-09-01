import nltk
from nltk.corpus import wordnet
from nltk import FreqDist, pos_tag
import pandas as pd
import fasttext
import gensim
#from gensim.models import fasttext
#import pickle
import pickle5 as pickle
from typing import List
import tensorflow as tf
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from pydantic.class_validators import validator


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')


from data_prep import cleaning
from model import model
from functions import data_for_model
import model_training

def load_models():
    """
    Loads fasttext model, tokenizer, gru model's weights to hold it in memory

    Returns:
        fasttext vectizer, tokenizer, gru model loaded with weights
    """
    physical_devices = tf.config.list_physical_devices('CPU')
    tf.config.experimental.set_visible_devices(physical_devices)

    # FastText model
    # fasttext_model = fasttext.load_model("train_test_all_cov_300_no_neg_final.bin")

    # Tokenizer
    with open(f'tokenizer_gru_cons_full_data_gen2.pickle', 'rb') as handle:
        tokenizer = pickle.load(file=handle)


    # fastext

    #ft = fasttext.load_model(path="train_test_all_cov_300_no_neg_final.bin")
    ft = gensim.models.KeyedVectors.load('FastText_v7/FastText_TrainData_v7.bin')


    # GRU model weights
    Model = model()
    gru_model = Model.set_model_params(tokenizer)
    gru_model.load_weights(f"gru_cons_full_data_gen2.h5")

    return tokenizer, gru_model,ft


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return 'v'


class NoteValidator(BaseModel):
    """
    Validation class for the incoming inspection note data
    """
    noteId : str
    noteText : str

    @validator("noteText")
    def check_note(cls, value):
        """
        Check the note if it is of type string
        """
        assert type(value) == str, "Error note is not a string"
        if len(value) < 1:
            raise ValueError(f"Note too small {value}")
        return value



class BodyValidator(BaseModel):
    """
    Validation class for the incoming inspection note data
    """
    inspection_data : List
    top_k = 1

    @validator("inspection_data")
    def check_note(cls, value):
        """
        Check the note if it is of type string
        """
        assert type(value) == list, "Error note is not a string"
        if len(value) == 0:
            raise ValueError(f"Please don't send invalid request")
        return value



# Server
app = FastAPI(debug=True)

# Load the models in memory
tokenizer, gru_model,ft= load_models()
clean = cleaning()
funcReq = data_for_model()
max_seq_len1 = 50
max_seq_len2 = 10

# Load the vectors
itemList = pd.read_csv('ItemListFinal.csv')
title_o = list(itemList['item'])
titles = list(itemList['cleaned_item'])
ids = list(itemList['item_id'])
item_id_dict = {}
item_title_dict = {}
for i in range(len(titles)):
    item_id_dict[titles[i]] = ids[i]
    item_title_dict[titles[i]] = title_o[i]

parentChild = pd.read_csv('Parent_child_items.csv')
item_parent_dict = {}
item_parentId_dict = {}
for i in range(len(parentChild)):
    item_parent_dict[parentChild['cleaned_item'][i]] = parentChild['Parent Item Title'][i]
    item_parentId_dict[parentChild['cleaned_item'][i]] = parentChild['Parent Item ID'][i]



@app.get("/hello")
async def check ():
    return "HelloWorld.."

@app.get("/train")
async def train_model ():
    data=pd.read_csv('data.csv')
    item_list = list(itemList['cleaned_item'])
    cleaned_data=clean.cleaner_for_whole_data(data)
    model,MRR_score=model_training.Model_training().model_selection(cleaned_data, max_seq_len1,max_seq_len2, ft,item_list,itemList)
    return MRR_score



@app.post("/")
async def create_item(body: BodyValidator):
    # Place holder
    data_dumper = []
    print("body")

    # Loop and collect
    for note in body.inspection_data:
        # Validation
        try:
            note = NoteValidator(**note)
        except:
            return "Error API request is not in good shape"

        cleanedText = clean.pre_process(note.noteText, get_wordnet_pos)
        item_list = list(itemList['cleaned_item'])
        rank_list = funcReq.rank_list(cleanedText, max_seq_len1, max_seq_len2, tokenizer, gru_model, item_list)

        # Local config time
        list_curr = []
        for i, (item, score) in enumerate(rank_list.items()):
            try:
                
                list_curr.append({
                    "ItemId" : str(item_id_dict[item]),
                    "ItemTitle" : str(item_title_dict[item]),
                    "Percentage" : float(score),
                    "ParentItemId" : str(item_parentId_dict[item]),
                    "ParentItemTitle" : str(item_parent_dict[item])
                    
                })
            
            except:
                
                list_curr.append({
                    "ItemId" : str(item_id_dict[item]),
                    "ItemTitle" : str(item_title_dict[item]),
                    "Percentage" : float(score),
                    "ParentItemId" : str(item_id_dict[item]),
                    "ParentItemTitle" : str(item_title_dict[item]),
                    
                })

            if i+1 == body.top_k:
                break

        
        # Global note dictionary
        note_dict = {
            "noteId" : note.noteId,
            "noteText" : note.noteText,
            "results" : list_curr,
        }

        data_dumper.append(note_dict)
    
    return data_dumper


@app.post("/")
async def calling(body: BodyValidator):
    data_dumper = await create_item(body)
    return data_dumper



