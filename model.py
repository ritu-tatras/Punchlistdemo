import pickle
import tensorflow as tf

tf.random.set_seed(20)

class model():

    def gru_encoder_arch_shared_embedding(self, vocab_size, vector_size, embedding_matrix, trainable=True, embeddding_pass=True):
        """
        Make the encoder architecture
        """
        # Make the input dimension
        input_text = tf.keras.layers.Input(shape=(50,), name="note_input")
        input_item = tf.keras.layers.Input(shape=(10,), name="item_input")
        
        # Shared Embedding
        if embeddding_pass:
            shared_embedding = tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=vector_size,
                                                        weights=[embedding_matrix], trainable=trainable)
        else:
            shared_embedding = tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=vector_size,
                                                        trainable=trainable)
        
        # Pass the model through the layer
        embed_text = shared_embedding(input_text)
        gru_text = tf.keras.layers.GRU(units=400, return_sequences=False)(embed_text)
        
        embed_item = shared_embedding(input_item)
        gru_item = tf.keras.layers.GRU(units=400, return_sequences=False)(embed_item)
        
        # Dot product the layers 
        dot = tf.keras.layers.Dot(axes=(-1, -1), normalize=True)([gru_text, gru_item])
        
        # Construct the model
        model = tf.keras.Model(inputs=[input_text, input_item], outputs=dot)
        return model

    def set_model_params(self, tokenizer):
        model = self.gru_encoder_arch_shared_embedding(vocab_size=len(tokenizer.word_index), embedding_matrix=None, embeddding_pass=False, trainable=False, vector_size=300)
        return model

