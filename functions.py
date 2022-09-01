from tensorflow.keras.preprocessing import sequence
import pandas as pd

class data_for_model():

    def prepare_token_ids(self, df, max_seq_len1, max_seq_len2, tokenizer):
        
        word_seq_text = tokenizer.texts_to_sequences(list(df['cleaned_note_text']))
        word_seq_item = tokenizer.texts_to_sequences(list(df['cleaned_item_text']))

        word_seq_text = sequence.pad_sequences(word_seq_text, maxlen=max_seq_len1)
        word_seq_item = sequence.pad_sequences(word_seq_item, maxlen=max_seq_len2)

        return word_seq_text, word_seq_item

    def rank_list(self, text, max_seq_len1, max_seq_len2, tokenizer, model, item_list):
        item_scores = {}
        df = pd.DataFrame()
        df['cleaned_item_text'] = item_list
        df['cleaned_note_text'] = text
        input_tokens = self.prepare_token_ids(df, max_seq_len1, max_seq_len2, tokenizer)
        pred = model.predict(input_tokens)
        for i in range(len(pred)):
            item_scores[item_list[i]] = pred[i][0]

        itemScores = sorted(item_scores.items(), key=lambda x: x[1])
        itemScores.reverse()

        return dict(itemScores)
