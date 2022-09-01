import os
import nltk
import scipy
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm.notebook import tqdm
from nltk.corpus import stopwords
import en_core_web_sm
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


GPU_to_use=1
gpus=tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[GPU_to_use], True)
# tf.config.experimental.set_visible_devices(gpus[GPU_to_use], 'GPU')


tqdm.pandas()
lemma = nltk.wordnet.WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer("english")
nlp = en_core_web_sm.load()
#nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
stops = set(stopwords.words('english'))

from functions import data_for_model

funcReq = data_for_model()



def sent_vec(sent, ft):
    v = []
    b = sent.split()
    for j in range(len(b)):
        wv = ft.wv[b[j]]
        v.append(wv)
    u = np.array(v)
    s = u.sum(axis=0)
    avg = s/len(u)
    return avg





def silhouetteAnalyis(X, numberOfClusters):
    """
       calculate the optimal number of clusters
       """
    silhouette_score_values = []

    for i in numberOfClusters:
        try:
            classifier = KMeans(i, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0,
                                random_state=None, copy_x=True)
            classifier.fit(X)
            labels = classifier.predict(X)
            silhouette_score_values.append(metrics.silhouette_score(X, labels, metric='euclidean',
                                                                    sample_size=None, random_state=None))
        except:
            pass

    Optimal_NumberOf_Components = numberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
    return Optimal_NumberOf_Components




def get_negative_titles(Idx, text, model, item_list, item_list_vec, train1s, tokenizer, ft,
                        max_seq_len1=50, max_seq_len2=10):
    """
       Will generate the negative instances for the data
       based on embeddings
       """
    score = funcReq.rank_list(text, max_seq_len1, max_seq_len2, tokenizer, model, item_list)
    index = list(score.keys()).index(train1s['cleaned_item_text'][Idx])
    confidence = list(score.values())[index]
    items_model_mapped = list(score.keys())
    items_required = items_model_mapped[:index]

    if len(items_required) > 5:
        cosineSimilarity = []
        text_vec = sent_vec(text, ft)
        for i in items_required:
            item_vec = item_list_vec[i]
            cosine_sim = 1 - scipy.spatial.distance.cosine(text_vec, item_vec)
            cosineSimilarity.append(cosine_sim)

        cosineSimilarity_2D = np.array(cosineSimilarity)
        cosineSimilarity_2D = cosineSimilarity_2D.reshape(-1, 1)

        numberOfClusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        n_clusters = silhouetteAnalyis(cosineSimilarity_2D, numberOfClusters)
        sc = SpectralClustering(n_clusters=n_clusters).fit(cosineSimilarity_2D)
        SpectralClustering(affinity='rbf', assign_labels='kmeans', coef0=1, degree=3, eigen_solver=None,
                           eigen_tol=0.0, gamma=1.0, kernel_params=None, n_clusters=n_clusters, n_components=None,
                           n_init=10, n_jobs=None, n_neighbors=10, random_state=None)
        labels = sc.labels_
        all_indices = {}
        for i in np.unique(labels):
            all_indices[i] = [k for k in range(len(labels)) if labels[k] == i]

        title_ix = []
        for j in list(all_indices.keys()):
            indx = np.random.randint(0, len(all_indices[j]))
            title_ix.append(all_indices[j][indx])

        negative_titles = []
        for g in title_ix:
            negative_titles.append(items_required[g])


    else:
        negative_titles = items_required
        n_clusters = len(items_required)

    return negative_titles, n_clusters




def create_dataframe_with_negative_samples(addedText,negativeTitles,original_data,itemListDf):

    """create dataframe to add negative samples with original data"""
    # TEXT

    print(original_data,"original data")

    negSamp = pd.DataFrame()
    negSamp['cleaned_note_text'] = addedText
    negSamp['cleaned_item_text'] = negativeTitles



    text_orig_cleaned_dict = {}
    for i in original_data.index:
        print(i,"itr")
        print(original_data['cleaned_note_text'][i],"data")
        print(original_data['note_text'][i])
        text_orig_cleaned_dict[original_data['cleaned_note_text'][i]] = original_data['note_text'][i]

    # TEXT-ID
    text_id_dict = {}
    for i in original_data.index:
        text_id_dict[original_data['cleaned_note_text'][i]] = original_data['note_id'][i]

    # ITEM
    item_orig_cleaned_dict = {}
    for i in itemListDf.index:
        item_orig_cleaned_dict[itemListDf['cleaned_item'][i]] = itemListDf['item'][i]

    # ITEM-ID
    item_id_dict = {}
    for i in itemListDf.index:
        item_id_dict[itemListDf['cleaned_item'][i]] = itemListDf['item_id'][i]

    # SOURCE
    source_dict = {}
    for i in original_data.index:
        source_dict[original_data['cleaned_note_text'][i]] = original_data['data_source'][i]

    # TEXT
    ns_text_orig = []
    for i in negSamp.index:
        ns_text_orig.append(text_orig_cleaned_dict[negSamp['cleaned_note_text'][i]])

    # TEXT-ID
    ns_text_id = []
    for i in negSamp.index:
        ns_text_id.append(text_id_dict[negSamp['cleaned_note_text'][i]])

    # ITEM
    ns_item_orig = []
    for i in negSamp.index:
        ns_item_orig.append(item_orig_cleaned_dict[negSamp['cleaned_item_text'][i]])

    # ITEM-ID
    ns_item_id = []
    for i in negSamp.index:
        ns_item_id.append(item_id_dict[negSamp['cleaned_item_text'][i]])

    # SOURCE
    ns_source = []
    for i in negSamp.index:
        ns_source.append(source_dict[negSamp['cleaned_note_text'][i]])

    negSamp['note_text'] = ns_text_orig
    negSamp['item_text'] = ns_item_orig
    negSamp['note_id'] = ns_text_id
    negSamp['item_id'] = ns_item_id
    negSamp['data_source'] = ns_source
    negSamp['label'] = 0

    negSamp = negSamp[['note_text', 'note_id', 'item_text', 'item_id', 'label', 'data_source', 'cleaned_note_text', 'cleaned_item_text']]
    print(negSamp.shape)

    combined = original_data.append(negSamp).sample(frac=1).reset_index(drop=True)
    print(combined.shape)

    combined.to_csv('combined_data_with_negative_samples.csv', index=False)
    negSamp.to_csv('negative_samples_gen.csv', index_label=False)
    return combined



def genrate_negative_samples(data,model,item_list,tokenizer,ft,itemListDf):
    negativeTitles = []
    addedText = []
    clusters_count = []
    EXCEPTIONS = []

    train1s = data[data['label'] == 1]
    train_1s_text = list(train1s['cleaned_note_text'])
    train1s.reset_index(drop=True, inplace=True)

    item_list_vec = {}
    for i in item_list:
        item_list_vec[i] = sent_vec(i, ft)

    for Inx in train1s.index:
        try:
            #with tf.device('/CPU:1'):
            neg_t, c = get_negative_titles(Inx, train1s['cleaned_note_text'][Inx],
                                           model, item_list,
                                           item_list_vec,
                                           train1s,
                                           tokenizer,
                                           ft)
            negativeTitles.extend(neg_t)
            aT = [train_1s_text[Inx]]*len(neg_t)
            addedText.extend(aT)
            clusters_count.append(c)
        except Exception as e:
            EXCEPTIONS.append(Inx)
            print(e)

    data_with_negative_instances=create_dataframe_with_negative_samples(addedText, negativeTitles, data, itemListDf)

    return data_with_negative_instances
