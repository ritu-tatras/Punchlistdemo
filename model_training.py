import random
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer


from functions import data_for_model
import negative_sample_generation
data_func=data_for_model()
tqdm.pandas()



def create_independent_splits(full_data, frac_ones=0.05, frac_twos=0.15):
    """
    Create independent datasets based on the inspection notes
    """
    # Init the data
    notes_with_freq_one = []
    notes_with_freq_two = []
    notes_with_freq_n = []

    distribution = full_data["cleaned_note_text"].value_counts().reset_index()
    distribution.columns = ["cleaned_note_text", "count"]
    notes_with_freq_one = list(set(distribution[distribution["count"] == 1]["cleaned_note_text"].values.tolist()))
    notes_with_freq_two = list(set(distribution[distribution["count"] == 2]["cleaned_note_text"].values.tolist()))
    notes_with_freq_n = list(set(distribution[distribution["count"] > 2]["cleaned_note_text"].values.tolist()))

    print(f"One Freq : {len(notes_with_freq_one)}")
    print(f"Two Freq : {len(notes_with_freq_two)}")

    for _ in range(4):
        random.shuffle(notes_with_freq_one)
        random.shuffle(notes_with_freq_two)

    # Index
    index_test_ones = int(len(notes_with_freq_one) * frac_ones)
    ones_test = notes_with_freq_one[:index_test_ones]
    ones_train = notes_with_freq_one[index_test_ones:]

    index_test_twos = int(len(notes_with_freq_two) * frac_twos)
    twos_test = notes_with_freq_two[:index_test_twos]
    twos_train = notes_with_freq_two[index_test_twos:]

    test_data = ones_test + twos_test
    train_data = ones_train + twos_train + notes_with_freq_n
    unique_notes = list(set(full_data["cleaned_note_text"].unique().tolist()))

    assert len(set(train_data).intersection(set(test_data))) == 0, "Commons Found"
    assert len(set(test_data).union(set(train_data))) == len(unique_notes), f"Missing Data : {len(set(test_data).union(set(train_data)))} | {len(unique_notes)}"

    cols = full_data.columns
    test_df = pd.DataFrame(columns=cols)
    bar = test_data
    for in_text_curr in bar:
        df_curr = full_data[full_data["cleaned_note_text"] == in_text_curr]
        test_df = test_df.append(df_curr[cols])
    cols = full_data.columns
    train_df = pd.DataFrame(columns=cols)
    bar = train_data
    for in_text_curr in bar:
        df_curr = full_data[full_data["cleaned_note_text"] == in_text_curr]
        train_df = train_df.append(df_curr[cols])


    return train_df, test_df


def train_test_split(data):
    """
    to split data into train and test
    input :   full data
    output: train dataframe  and test dataframe"""
    train_df, test_df = create_independent_splits(full_data=data)
    train_df.reset_index(drop=True).to_csv("data_train.csv", index=False)
    test_df.reset_index(drop=True).to_csv("data_test.csv", index=False)
    return train_df,test_df

class Model_training:

    def generate_class_weights(self,class_series, multi_class=True, one_hot_encoded=False):
        if multi_class:
            # If class is one hot encoded, transform to categorical labels to use compute_class_weight
            if one_hot_encoded:
                class_series = np.argmax(class_series, axis=1)

            # Compute class weights with sklearn method
            class_labels = np.unique(class_series)
            print(class_labels,class_series,"label and series")

            class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
            return dict(zip(class_labels, class_weights))

        else:
            # It is neccessary that the multi-label values are one-hot encoded
            mlb = None
            if not one_hot_encoded:
                mlb = MultiLabelBinarizer()
                class_series = mlb.fit_transform(class_series)

            n_samples = len(class_series)
            n_classes = len(class_series[0])

            # Count each class frequency
            class_count = [0] * n_classes
            for classes in class_series:
                for index in range(n_classes):
                    if classes[index] != 0:
                        class_count[index] += 1

            # Compute class weights using balanced method
            class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
            class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
            return dict(zip(class_labels, class_weights))

    def sent_vec(self, sent, ft):
        v = []
        b = sent.split()
        for j in range(len(b)):
            wv = ft.wv[b[j]]
            v.append(wv)
        u = np.array(v)
        s = u.sum(axis=0)
        avg = s / len(u)
        return avg


    def tokenizer_func(self,data,ft):
        MAX_NB_WORDS = 8500
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
        #changed here add text in end
        tokenizer.fit_on_texts(list(data['cleaned_note_text'].unique()) + list(data['cleaned_item_text'].unique()))
        word_index = tokenizer.word_index
        embeddings_index = {}
        for i in word_index:
            embeddings_index[i] = self.sent_vec(i, ft)

        embed_dim = 300
        #changed min to max
        nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
        embedding_matrix = np.zeros((nb_words, embed_dim))

        words_not_found = []
        for word, i in word_index.items():
            if i >= nb_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                words_not_found.append(word)

        return embedding_matrix,len(word_index),tokenizer



    def rmse(self,y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def gru_encoder_arch_shared_embedding(self, input_length1, input_length2, vocab_size, vector_size, embedding_matrix,
                                          trainable=True):
        """
           Make the encoder architecture
           """
        # Make the input dimension
        input_text = tf.keras.layers.Input(shape=(input_length1,), name="note_input")
        input_item = tf.keras.layers.Input(shape=(input_length2,), name="item_input")

        # Shared Embedding
        #for temporary delete it later
        vector_size=300
        shared_embedding = tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=vector_size,
                                                     weights=[embedding_matrix], trainable=trainable)

        # Pass the model through the layer
        embed_text = shared_embedding(input_text)
        gru_text = tf.keras.layers.GRU(units=400, return_sequences=False)(embed_text)

        embed_item = shared_embedding(input_item)
        gru_item = tf.keras.layers.GRU(units=400, return_sequences=False)(embed_item)

        # Dot product the layers
        dot = tf.keras.layers.Dot(axes=(-1, -1), normalize=True)([gru_text, gru_item])

        # Construct the model
        model = tf.keras.Model(inputs=[input_text, input_item], outputs=dot)
        model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=[self.rmse])
        return model




    def model_training(self,train,test,max_seq_len1,max_seq_len2,ft):

        train = train.sample(frac=1)
        test = test.sample(frac=1)

        train.dropna(inplace=True)
        train.reset_index(drop=True, inplace=True)
        test.dropna(inplace=True)
        test.reset_index(drop=True, inplace=True)

        cls = list(list(train['label']))
        print(train['label'])
        print(cls,"class")
        cls_wts = self.generate_class_weights(cls, multi_class=True, one_hot_encoded=False)

        y_train = list(train['label'])
        y_test = list(test['label'])
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        embeddings_matrix,vocab_size,tokenizer=self.tokenizer_func(train,ft)
        word_seq_text, word_seq_item = data_func.prepare_token_ids(train, max_seq_len1, max_seq_len2, tokenizer)
        word_seq_text_test,word_seq_item_test= data_func.prepare_token_ids(test, max_seq_len1, max_seq_len2, tokenizer)


        model = self.gru_encoder_arch_shared_embedding(max_seq_len1, max_seq_len2, vocab_size, 400, embeddings_matrix)
        model.summary()
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                                                           verbose=0, min_delta=0.0001, mode='min')

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=4,
                                                          verbose=1, min_delta=0.0001, restore_best_weights=True)

        hist = model.fit((word_seq_text, word_seq_item), y_train, epochs=100, batch_size=32, validation_split=0.2,
                         callbacks=[lr_callback, early_stopping],
                         class_weight=cls_wts, verbose=1)

        model.evaluate((word_seq_text_test, word_seq_item_test), y_test, batch_size=64)

        model.save('GRU3_400_gen123.h5')

        return model,tokenizer

    def calculate_MRR(self,testdata,max_seq_len1, max_seq_len2, tokenizer, model, item_list):
        """

        calculate MRR Score for Testdata
        """

        ##########################################################################################
        test1s = testdata[testdata['label'] == 1]
        test1s.dropna(inplace=True)
        test1s.reset_index(drop=True, inplace=True)
        test1s['label'].value_counts()

        notFound = []
        indices = []
        for i in range(len(test1s)):
            score = data_func.rank_list(test1s['cleaned_note_text'][i], max_seq_len1, max_seq_len2, tokenizer, model, item_list)
            try:
                index = list(score.keys()).index(test1s['cleaned_item_text'][i])
                indices.append(index)

            except:
                notFound.append(i)

        print('indices:', len(indices), ' notFound:', len(notFound))
        reciprocal_ranks = []
        for i in indices:
            rr = round((1 / (i + 1)), 2)
            reciprocal_ranks.append(rr)

        MRR=(sum(reciprocal_ranks)) / (len(reciprocal_ranks))

        print('Mean Reciprocal Rank: ', (sum(reciprocal_ranks)) / (len(reciprocal_ranks)))


        return MRR


    def iteration_for_negative_sample_generations(self,train_df,test_df,tokenizer, max_seq_len1,max_seq_len2, ft,item_list,itemList,model):
        """function to generate negative samples and calculate MRR Score
        return MRR score , trained model and new train dataset
        """
        negative_samples = negative_sample_generation.genrate_negative_samples(train_df, model, item_list,tokenizer, ft, itemList)
        train_with_negative_samples = train_df.append(negative_samples)
        print(train_with_negative_samples,"traindata with negative sample")
        print(negative_samples,"negative samples")
        trained_model_with_negsamples,tokenizer = self.model_training(train_with_negative_samples, test_df,max_seq_len1, max_seq_len2, ft)
        MRR_Score_for_negsamples_model = self.calculate_MRR(test_df, max_seq_len1, max_seq_len2, tokenizer,trained_model_with_negsamples, item_list)

        return MRR_Score_for_negsamples_model,trained_model_with_negsamples,train_with_negative_samples,tokenizer


    def recursion_for_models(self,MRR_Score_for_base_model,MRR_Score_for_negsample1,base_trained_model,model_for_negative_sample,train_df, test_df, tokenizer,
                                                                                      max_seq_len1, max_seq_len2,ft, item_list, itemList):
        """
        recursive function to select generation of negative samples which gives highest MRR score
        :param MRR_Score_for_base_model:
        :param MRR_Score_for_negsample1:
        :param base_trained_model:
        :param model_for_negative_sample:
        :param train_df:
        :param test_df:
        :param tokenizer:
        :param max_seq_len1:
        :param max_seq_len2:
        :param ft:
        :param item_list:
        :param itemList:
        :return:  selected model and MRR Score
        """
        if MRR_Score_for_base_model > MRR_Score_for_negsample1:
            return base_trained_model,MRR_Score_for_base_model
        else:
            MRR_Score_for_base_model = MRR_Score_for_negsample1
            base_trained_model = model_for_negative_sample
            print(train_df,"train df")
            print(test_df,"testdf")

            MRR_Score_for_negsample1,model_for_negative_sample,train_data_with_negative_samples = self.iteration_for_negative_sample_generations(train_df, test_df, tokenizer,
                                                                                      max_seq_len1, max_seq_len2,
                                                                                      ft, item_list, itemList,
                                                                                      base_trained_model)
            train_df=train_data_with_negative_samples


            return self.recursion_for_models(MRR_Score_for_base_model,MRR_Score_for_negsample1,base_trained_model,model_for_negative_sample,train_df, test_df, tokenizer,
                                                                                      max_seq_len1, max_seq_len2,ft, item_list, itemList)



    def model_selection(self,data, max_seq_len1,max_seq_len2, ft,item_list,itemList):
        """
        To select model with generation of negative samples
        :param data: cleaned data
        :param tokenizer: tokenizer
        :param max_seq_len1: max sequence length for inspection note
        :param max_seq_len2: max sequence length for items
        :param ft: faste txt model
        :param item_list: list of items
        :param itemList: Dataframe for itemlist
        :return: Finalised model and MRR Score for test data
        """


        train_df, test_df = train_test_split(data)
        base_trained_model,tokenizer = self.model_training(train_df, test_df, max_seq_len1,max_seq_len2, ft)
        MRR_Score_for_base_model = self.calculate_MRR(test_df, max_seq_len1, max_seq_len2, tokenizer,base_trained_model, item_list)

        MRR_Score_for_negsample1,model_for_negative_sample,train_data_with_negative_samples,tokenizer=self.iteration_for_negative_sample_generations(train_df,test_df,tokenizer, max_seq_len1,max_seq_len2, ft,item_list,itemList,base_trained_model)


        final_model,MRR_Score=self.recursion_for_models(MRR_Score_for_base_model,MRR_Score_for_negsample1,base_trained_model,model_for_negative_sample,train_data_with_negative_samples, test_df, tokenizer,
                                                                                  max_seq_len1, max_seq_len2,ft, item_list, itemList)

        return final_model,MRR_Score














