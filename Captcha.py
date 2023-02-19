import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras

# Path to data directory
data_dir = Path('train')

# Get list of all the images
images = list(map(str, data_dir.rglob('*.png')))
labels = [img.split(os.path.sep)[-1].split('*.png')[0] for img in images]
char_counts = Counter(''.join(labels))
characters = set(char_counts.keys())

print('Number of Images Found: ', len(images))
print('Number of Labels Found: ', len(labels))
print('Number of Unique Characters: ', len(characters))
print('Characters Present: ', characters)

# Batch Size for Training and Validation
batch_size = 16

# Desired Image Dimensions
image_width = 200
image_height = 50

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two convolutional blocks and each will have
# a pooling layer which downsample the features by a factor of 2
# hence total downsampling factor would be 4
downsample_factor = 4

# Maximum length of any captcha in the dataset
max_length = max([len(label) for label in labels])

# Mapping characters to integers
char_to_num = keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.9, shuffle=True):
    size = len(images)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


# splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


def encode_single_sample(img_path, label):
    # Read Image and convert to float
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Normalize the image
    img = tf.image.resize(img, [image_height, image_width])
    # Split the image into max length number of tensors
    img = tf.transpose(img, perm=[1, 0, 2])
    # Encode the label
    label = char_to_num(tf.strings.unicode_split(label, input_encoding='UTF-8'))
    # Return the image and the label
    return {"image": img, "label": label}


# Create Dataset Objects

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset
    .map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .shuffle(buffer_size=len(x_train))
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    .cache()
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

# Visualize the Data

_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    images = batch["image"]
    labels = batch["label"]
    for i in range(16):
        img = (images[i] * 255).numpy().astype('uint8')
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode('utf-8')
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap='gray')
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis('off')
    plt.show()


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # compute the training-time los value and add it to the layer using 'self.add_loss()'
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_true)[1], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)


def build_model():
    # Inputs to the model
    input_img = keras.layers.Input(
        shape=(image_width, image_height, 1), name='images', dtype='float32'
    )
    labels = keras.layers.Input(name='label', shape=(None), dtype='float32')

    # First Conv Block
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        kernal_initializer='he_normal',
        padding='same',
        name='Conv1',
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name='pool1')(x)

    # Second Conv Block
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        kernal_initializer='he_normal',
        padding='same',
        name='Conv2',
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name='pool2')(x)

    # We have used two max pool with pool size and stride 2.
    # Hence, downsampled feature maps are 4x smaller.
    # The number of filters in the last layer is 64.
    # reshape accordingly before passing the output to the RNN part of the model
    new_shape = ((image_width // 4), (image_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name='reshape')(x)
    x = keras.layers.Dense(64, activation='relu', name='densel')(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_state=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_state=True, dropout=0.25))(x)

    # Output Layer
    x = keras.layers.Dense(len(characters) + 1, activation='softmax', name='dense2')(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name='ctc_loss')(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], output=output, name='ocr_model_v1'
    )

    # Optimizer
    opt = keras.optimizers.Adam()
    # compile the model and return
    model.compile(optimizer=opt)
    return model

# Get the model
model = build_model()
model.summary()


# Train the neural net
epochs = 100
early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True
)

# train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)

# Predictions with trained Neural Network
# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name='image').input, model.get_layer(name='dense2').output
)

#plot the loss of the model during training
plt.plot(history.history['loss'])
plt.plot(history.histroy['val_loss'])
plt.xticks(range(0, epochs+1,10))
plt.title("OCR Model")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()