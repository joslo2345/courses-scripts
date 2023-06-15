# Setup
import os
import time
import numpy as np
import tensorflow as tf

# Download the shakespeare dataset

### you can modify this line with any other text you wanna try 
path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)

# Read the data

### download the file and then decode
text = open(path_to_file, 'rb').read().decode(encoding="utf-8")
print(f"Length of text: {len(text)} characters")

### print the first 250 characters
print(text[:250])

### print the the ammount of unique characters are in the corpus/document
vocab = sorted(set(text))
print(f"{len(vocab)} unique characters")

# Process the Test

## vectorize the text 

"""
Before the training we need to convert the strings to a numerical representation.
Using tf.keras.layers.StringLookup layer can convert each character into a numeric ID. It just needs the text to be split into tokens first.
"""

example_texts = ["abcdefg", "xyz"]

# TODO 1
chars = tf.strings.unicode_split(example_texts, input_encoding="UTF-8")
print(chars)

### Now create the tf.keras.layers.StringLookup layer
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)

### It converts from tokens to character IDs
ids = ids_from_chars(chars)
print(ids)

"""
Since the goal of this tutorial is to generate text, it will also be important to invert this representation and 
recover human-readable strings from it. For this you can use tf.keras.layers.StringLookup(..., invert=True).

Note: Here instead of passing the original vocabulary generated with sorted(set(text)) use the get_vocabulary() 
method of the tf.keras.layers.StringLookup layer so that the [UNK] tokens is set the same way.
"""
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)

### This layer recovers the characters from the vectors of IDs, and returns them as a tf.RaggedTensor of characters:
chars = chars_from_ids(ids)
print(chars)

### You can tf.strings.reduce_join to join the characters back into strings.
tf.strings.reduce_join(chars, axis=-1).numpy()

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

"""
# The prediction task
Given a character, or a sequence of characters, what is the most probable next character? This is the task you're training the model to perform. The input to the model will be a sequence of characters, and you train the model to predict the output—the following character at each time step.

Since RNNs maintain an internal state that depends on the previously seen elements, given all the characters computed until this moment, what is the next character?

# Create training examples and targets
Next divide the text into example sequences. Each input sequence will contain seq_length characters from the text.

For each input sequence, the corresponding targets contain the same length of text, except shifted one character to the right.

So break the text into chunks of seq_length+1. For example, say seq_length is 4 and our text is "Hello". The input sequence would be "Hell", and the target sequence "ello".

First use the tf.data.Dataset.from_tensor_slices function to convert the text vector into a stream of character indices.
"""

# TODO 2
all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))
all_ids

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode("utf-8"))

seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

### The batch method lets you easily convert these individual characters to sequences of the desired size.
sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

for seq in sequences.take(1):
    print(chars_from_ids(seq))

### It's easier to see what this is doing if you join the tokens back into strings:
for seq in sequences.take(5):
    print(text_from_ids(seq).numpy())

"""
For training you'll need a dataset of (input, label) pairs. Where input and label are sequences. At each time step the input is 
the current character and the label is the next character.

Here's a function that takes a sequence as input, duplicates, and shifts it to align the input and label for each timestep:
"""

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

split_input_target(list("Tensorflow"))

dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())

"""
Create training batches
You used tf.data to split the text into manageable sequences. 
But before feeding this data into the model, you need to shuffle the data and pack it into batches.
"""


# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset.shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

print(dataset)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        # TODO - Create an embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # TODO - Create a GRU layer
        self.gru = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True
        )
        # TODO - Finally connect it with a dense layer
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        # since we are training a text generation model,
        # we use the previous state, in training. If there is no state,
        # then we initialize the state
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(
        example_batch_predictions.shape,
        "# (batch_size, sequence_length, vocab_size)",
    )

model.summary()

sampled_indices = tf.random.categorical(
    example_batch_predictions[0], num_samples=1
)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

print(sampled_indices)

print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())

# TODO - add a loss function here
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
print(
    "Prediction shape: ",
    example_batch_predictions.shape,
    " # (batch_size, sequence_length, vocab_size)",
)
print("Mean loss:        ", example_batch_mean_loss)

tf.exp(example_batch_mean_loss).numpy()

model.compile(optimizer="adam", loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = "./training_checkpoints"
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True
)

EPOCHS = 10

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(["[UNK]"])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float("inf")] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())],
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(
            inputs=input_ids, states=states, return_state=True
        )
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(["ROMEO:"])
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model.generate_one_step(
        next_char, states=states
    )
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode("utf-8"), "\n\n" + "_" * 80)
print("\nRun time:", end - start)

start = time.time()
states = None
next_char = tf.constant(["ROMEO:", "ROMEO:", "ROMEO:", "ROMEO:", "ROMEO:"])
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model.generate_one_step(
        next_char, states=states
    )
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result, "\n\n" + "_" * 80)
print("\nRun time:", end - start)

tf.saved_model.save(one_step_model, "one_step")
one_step_reloaded = tf.saved_model.load("one_step")

states = None
next_char = tf.constant(["ROMEO:"])
result = [next_char]

for n in range(100):
    next_char, states = one_step_reloaded.generate_one_step(
        next_char, states=states
    )
    result.append(next_char)

print(tf.strings.join(result)[0].numpy().decode("utf-8"))

class CustomTraining(MyModel):
    @tf.function
    def train_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return {"loss": loss}

model = CustomTraining(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

model.fit(dataset, epochs=1)

EPOCHS = 10

mean = tf.metrics.Mean()

for epoch in range(EPOCHS):
    start = time.time()

    mean.reset_states()
    for batch_n, (inp, target) in enumerate(dataset):
        logs = model.train_step([inp, target])
        mean.update_state(logs["loss"])

        if batch_n % 50 == 0:
            template = (
                f"Epoch {epoch+1} Batch {batch_n} Loss {logs['loss']:.4f}"
            )
            print(template)

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print()
    print(f"Epoch {epoch+1} Loss: {mean.result().numpy():.4f}")
    print(f"Time taken for 1 epoch {time.time() - start:.2f} sec")
    print("_" * 80)

model.save_weights(checkpoint_prefix.format(epoch=epoch))