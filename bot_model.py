import tensorflow as tf
import unicodedata
import re
import os
import pickle


class Attention(tf.keras.Model):
    """Layer that performs Bahdanau Attention."""
    def __init__(self, units):
        """Units (int) - input dimension."""

        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


class Encoder(tf.keras.Model):
    """Encoder layer of s2s model."""

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        """vocab_size (int) - Vocabulary size
        embedding_dim (int) - Embedding layer dimension,
        enc_units (int) - Output dimension
        batch_size (int) - Batch size."""

        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        """Returns tf.tensor() initialized with zeroes and with size [batch_size, enc_units]."""

        return tf.zeros([self.batch_size, self.enc_units])


class Decoder(tf.keras.Model):
    """Decode layer of s2s model."""

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        """vocab_size (int) - Vocabulary size
        embedding_dim (int) - Embedding layer dimension,
        dec_units (int) - Output dimension
        batch_size (int) - Batch size."""

        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = Attention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state


class Bot:
    """Class that realize s2s model."""

    @staticmethod
    def prep_sentence(sentence):
        """Prepare sequence by removing special characters.
        sentence - string"""

        sentence = sentence.lower().strip()
        sentence = ''.join(c for c in unicodedata.normalize('NFD', sentence)
                           if unicodedata.category(c) != 'Mn')
        sentence = re.sub(r'http\S+', '', sentence)             # remove hyperlinks
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)       # separate punctuation
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)     # replace all other symbols with space
        sentence = sentence.rstrip().strip()
        if not sentence:
            return None
        sentence = '__go__ ' + sentence + ' __eos__'
        return sentence

    def __init__(self, vocab_size, embedding_dim, units, batch_size, max_input_length=0, max_output_length=0):
        """vocab_size (int)- Vocabulary size
        embedding_dim (int)- Embedding layer dimension,
        units (int)- Encoder and Decoder output dimensions,
        batch_size (int)- Batch size,
        max_input_length (int) and max_output_length (int) - Limits to input/output sequence."""

        self.vocab_size = vocab_size
        self.units = units
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.decoder = Decoder(vocab_size, embedding_dim, units, batch_size)
        self.encoder = Encoder(vocab_size, embedding_dim, units, batch_size)

        self.optimizer = tf.keras.optimizers.Adam()
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size+1, filters='', oov_token='__unk__'
        )
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)

    def save_model(self, path):
        """path - Path to directory where will be stored tokenizer and weights.
        Returns path to checkpoint."""

        with open(os.path.join(path, 'tokenizer.pickle'), 'wb') as file:
            pickle.dump(self.tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)
        path = self.checkpoint.save(file_prefix=os.path.join(path, 'checkpoint'))
        return path

    def load_model(self, path, checkpoint_path):
        """path - Path to directory where was stored tokenizer and weights."""

        with open(os.path.join(path, 'tokenizer.pickle'), 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.checkpoint.restore(checkpoint_path)

    def loss_function(self, correct, predicted):
        """Calculates loss function."""

        mask = tf.math.logical_not(tf.math.equal(correct, 0))
        loss_ = self.loss_object(correct, predicted)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, inputs, target, hidden):
        """Executes 1 batch in training loop."""

        loss = 0
        with tf.GradientTape() as tape:
            enc_output, hidden = self.encoder(inputs, hidden)
            dec_input = tf.expand_dims([self.tokenizer.word_index['__go__']] * self.batch_size, 1)
            for t in range(1, target.shape[1]):
                predictions, hidden = self.decoder(dec_input, hidden, enc_output)
                loss += self.loss_function(target[:, t], predictions)
                dec_input = tf.expand_dims(target[:, t], 1)
        batch_loss = (loss / int(target.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss, predictions

    def train(self, epochs, inputs, outputs):
        """epochs (int) - Number of epochs,
        inputs, outputs - Lists of sequences(strings)."""

        self.tokenizer.fit_on_texts(inputs + outputs)
        with open('tokenizer.pickle', 'wb') as file:
            pickle.dump(self.tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

        inputs = self.tokenizer.texts_to_sequences(inputs)
        self.max_input_length = max(self.max_input_length, max(len(s) for s in inputs))
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=self.max_input_length, padding='post')

        outputs = self.tokenizer.texts_to_sequences(outputs)
        self.max_output_length = max(self.max_output_length, max(len(s) for s in outputs))
        outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs, maxlen=self.max_output_length, padding='post')

        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        steps_per_epoch = len(inputs) // self.batch_size
        for epoch in range(epochs):
            print('Staring epoch {}'.format(epoch + 1))
            hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (input, output)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss, predictions = self.train_step(input, output, hidden)
                total_loss += batch_loss

                if batch % 10 == 0:
                    print('Batch {} Loss {:.4f}'.format(batch, batch_loss.numpy()))

                if batch % 50 == 0:
                    self.checkpoint.save('./checkpoints/checkpoint')
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))

    def evaluate(self, sentence, hidden):
        """Predicts answer for given sentence.
        Can only work if batch_size = 1.
        sentence - string
        hidden - tf.tensor()"""

        assert self.batch_size == 1
        sentence = Bot.prep_sentence(sentence)
        if sentence is None:
            raise Exception('Empty string')
        inputs = self.tokenizer.texts_to_sequences([sentence]);
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                               maxlen=self.max_input_length,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)
        enc_output, hidden = self.encoder(inputs, hidden)
        dec_input = tf.expand_dims([self.tokenizer.word_index['__go__']], 0)
        result = ''
        for t in range(self.max_output_length):
            predictions, hidden = self.decoder(dec_input, hidden, enc_output)
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.tokenizer.index_word[predicted_id + 1] + ' '
            if self.tokenizer.index_word[predicted_id + 1] == '__eos__':
                return result.replace('__go__', '').replace('__eos__', '').strip(), hidden
            dec_input = tf.expand_dims([predicted_id], 0)
        return result.replace('__go__', '').replace('__eos__', '').strip(), hidden


if __name__ == '__main__':
    inp_seq_file = '/reddit/inp.txt'
    out_seq_file = '/reddit/out.txt'
    vocab_size = 20000
    embedding_dim = 128
    units = 512
    batch_size = 2
    epochs = 10

    print('Reading files...')
    inputs = open(inp_seq_file, 'r', encoding='utf-8').read().strip().split('\n')
    outputs = open(out_seq_file, 'r', encoding='utf-8').read().strip().split('\n')

    inputs = inputs[:3500]      # cut dataset
    outputs = outputs[:3500]

    model = Bot(vocab_size, embedding_dim, units, batch_size)
    print('Starting training...')
    model.train(epochs, inputs, outputs)
