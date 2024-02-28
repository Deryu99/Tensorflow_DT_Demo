import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

dataset_name = 'beans'

# Iterations over the entire dataset
epochs = 10

# Function takes a dataset, split ('train', 'valid', or 'test'), and batch size as arguments.
def wrangle_data_GenPlus(dataset, split, bs):
    # Casts the image to float32 and normalizes it by dividing by 255.
    wrangled = dataset.map(lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl))
    if split == 'train':
        features = np.array([x[0] for x in wrangled])
        labels = np.array([x[1] for x in wrangled])
        train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            zoom_range=0.2,
            rotation_range=20,
            fill_mode='nearest'
        )
        wrangled = train_data_gen.flow(features, labels, batch_size=bs)
    
    # Caches the elements in this dataset. loat it into the memory to go faster
    elif split in ('valid', 'test'):
        # If the split is 'valid' or 'test':
            # It caches the elements in the dataset for faster access.
            # It batches the elements into desired sizes.
            # It prefetches elements to overlap data loading with training.
        wrangled = wrangled.cache()
        wrangled = wrangled.batch(bs)  # Combines consecutive elements of this dataset into batches.
        wrangled = wrangled.prefetch(tf.data.AUTOTUNE)
    return wrangled

# Ensures the code below only runs when the script is executed directly (not imported as a module).
if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy() # Multiple GPUs
    batch_size = 64
    GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync
    
    with strategy.scope():
        # prepare the data
        test_ds, info = tfds.load(dataset_name, split='test', as_supervised=True, with_info=True, shuffle_files=True)
        valid_ds = tfds.load(dataset_name, split=f'validation', as_supervised=True)
        train_ds = tfds.load(dataset_name, split=f'train', as_supervised=True)

        # Wrangle each dataset (train, validation, and test) with appropriate split and batch size.
        train_data = wrangle_data_GenPlus(train_ds, 'train', bs=GLOBAL_BATCH_SIZE)
        valid_data = wrangle_data_GenPlus(valid_ds, 'valid', bs=GLOBAL_BATCH_SIZE)
        test_data = wrangle_data_GenPlus(test_ds, 'test', bs=GLOBAL_BATCH_SIZE)

        # Model definition:
            # Creates a sequential model (tf.keras.Sequential).
            # Defines the layers of the CNN model:
            # Input layer for images of size (500, 500, 3) (channels).
            # Resizes the input images to (125, 125).
            # Several convolutional layers with ReLU activation and MaxPooling.
            # Flattening layer to convert from 2D to 1D for dense layers.
            # Dense layers with ReLU activation for feature extraction.
            # Final dense layer with softmax activation for 3 output classes (potentially corresponding to different bean types).
        # Names the model as "cnn_model".
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer((500, 500, 3)),
            tf.keras.layers.experimental.preprocessing.Resizing(125, 125),
            tf.keras.layers.Conv2D(64, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax)
        ], name='cnn_model')

        # Compiles the model using the Adam optimizer
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])

    # Trains the model using the fit method:
            # Provides the training data (train_data).
            # Sets the validation data (valid_data) for monitoring performance during training.
            # Specifies the number of epochs for training (epochs).
    history = model.fit(train_data, validation_data=valid_data, epochs=epochs)
