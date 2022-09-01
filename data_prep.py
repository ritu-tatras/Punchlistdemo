import re
import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm
from nltk import pos_tag
from nltk.corpus import wordnet



def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return 'v'


class cleaning():

    # def get_wordnet_pos(self, word):
    #     """Map POS tag to first character lemmatize() accepts"""
    #     tag = pos_tag([word])[0][1][0].upper()
    #     tag_dict = {"J": wordnet.ADJ,
    #                 "N": wordnet.NOUN,
    #                 "V": wordnet.VERB,
    #                 "R": wordnet.ADV}
    #     return 'v'

    def pre_process(self, value):
        """
        simple clean operation
        all hyperparams are fixed here
        """
        # SIMPLE CLEANING
        value = str(value)
        value = value.replace("\n", " ")
        value = value.replace("(s)", " ")
        value = re.sub(pattern=r"\d", repl=r" ", string=value)
        rep_str = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~•.""'
        for i in rep_str:
            value = value.replace(str(i), " ")

        # SPACY CLEANING
        value = " ".join([str(i).strip() for i in nlp(value) if i.is_stop==False])   
        value = " ".join([(str(i)).strip() for i in value.split() if len(i) <= 15 and len(i) > 2])

        # Remove stop words
        value = value.lower()

        # Lemmatize
        lemma = WordNetLemmatizer()
        value = " ".join([lemma.lemmatize(i, get_wordnet_pos(i)).strip() for i in value.split()])


        # Return
        return str(value)

    def freq_based_cleaning(self,in_df):
        """
        Performs the frequency based cleaning
        1. To select the most frequent label for the data
        """
        global_df = pd.DataFrame(columns=in_df.columns)
        tqdm_bar = in_df.groupby(by=["cleaned_note_text", "cleaned_item_text"])
        for idx, df_curr in tqdm_bar:
            # Find the value counts
            val_count_curr = df_curr["label"].value_counts()
            choosen_label = val_count_curr.index[0]

            # We have the label here

            # Add the data to the dataframe
            df_chosen = df_curr[df_curr["label"] == choosen_label]
            global_df = global_df.append(df_chosen)



        return global_df

    def post_fixes(self,in_df, fix1=False):
        """
        Returns the eval items index
        """
        list_indexes = []
        for idx, row in in_df.iterrows():
            item_curr = row["cleaned_item_text"]
            text_curr = row["cleaned_note_text"]
            if fix1:
                if "eval" in item_curr or "exclude" in item_curr:
                    list_indexes.append(idx)
            if len(text_curr.split()) < 3:
                list_indexes.append(idx)

        print(f"Total bad text found : {len(list_indexes)}")
        return list(set(list_indexes))

    def cleaning_phase_two(self,in_df):
        """
        Cleaning focused for getting the final label 1
        and fixing the dups in negatives
        """
        m = 0
        # Seperate the positives and negatives
        fs_pos = in_df[in_df["label"] == 1].reset_index(drop=True)
        fs_neg = in_df[in_df["label"] == 0].reset_index(drop=True)

        # Work on the positives first
        pos_df_fixed = pd.DataFrame(columns=in_df.columns)
        tqdm_bar = fs_pos.groupby("cleaned_note_text")
        for _, df_curr in tqdm_bar:
            # Collec the value counts of the item
            val_count_curr = df_curr["cleaned_item_text"].value_counts()
            choosen_item = val_count_curr.index[0]

            # Check other condition
            if val_count_curr[0] == 1 and len(val_count_curr.index) > 1:
                m += 1
                choosen_item = df_curr[df_curr["inter_word"] == max(df_curr["inter_word"])].iloc[0]["cleaned_item_text"]

            # Add the data to the dataframe
            df_chosen = df_curr[df_curr["cleaned_item_text"] == choosen_item].iloc[0]
            pos_df_fixed = pos_df_fixed.append(df_chosen)


        # Drop the duplicates from the data
        print(f"fs_neg shape before : {fs_neg.shape}")
        fs_neg = fs_neg.drop_duplicates(subset=["cleaned_note_text", "cleaned_item_text"]).reset_index(drop=True)
        print(f"fs_neg shape after : {fs_neg.shape}")

        # Merge the data together
        combined_data = pos_df_fixed.append(fs_neg).reset_index(drop=True)
        print(f"full data shape : {combined_data.shape}")

        return combined_data

    def cleaner_for_whole_data(self,data):
        data["cleaned_note_text"] = data["note_text"].apply(self.pre_process)
        data["cleaned_item_text"] = data["item_text"].apply(self.pre_process)

        data["cleaned_note_text"] = data["cleaned_note_text"].astype(str)
        data["cleaned_item_text"] = data["cleaned_item_text"].astype(str)
        data["note_text"] = data["note_text"].astype(str)
        data["item_text"] = data["item_text"].astype(str)
        full_data_no_eval = data.copy()

        indexes = self.post_fixes(in_df=full_data_no_eval, fix1=True)
        full_data_no_eval = full_data_no_eval.drop(index=indexes)

        cleaned_df_no_eval = self.freq_based_cleaning(in_df=full_data_no_eval)
        cleaned_df_no_eval.to_csv("non_timestamp_freq_cleaned.csv", index=False)

        list_inter = []
        for _, row in cleaned_df_no_eval.iterrows():
            curr_text = row["cleaned_note_text"]
            curr_item = row["cleaned_item_text"]
            list_inter.append(len(set(curr_text.split()).intersection(set(curr_item.split()))))

        cleaned_df_no_eval["inter_word"] = list_inter

        final_data_no_eval = self.cleaning_phase_two(in_df=cleaned_df_no_eval)

        final_data_no_eval.to_csv("cleaned_data_final.csv", index=False)

        return final_data_no_eval

