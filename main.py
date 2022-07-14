import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

dataset_name = 'beans'
epochs = 10


def wrangle_data_GenPlus(dataset, split, bs):
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
    elif split in ('valid', 'test'):  # Caches the elements in this dataset. loat it into the memory to go faster
        wrangled = wrangled.cache()
        wrangled = wrangled.batch(bs)  # Combines consecutive elements of this dataset into batches.
        wrangled = wrangled.prefetch(tf.data.AUTOTUNE)
    return wrangled


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    batch_size = 64
    GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync
    with strategy.scope():
        # prepare the data
        test_ds, info = tfds.load(dataset_name, split='test', as_supervised=True, with_info=True, shuffle_files=True)
        valid_ds = tfds.load(dataset_name, split=f'validation', as_supervised=True)
        train_ds = tfds.load(dataset_name, split=f'train', as_supervised=True)

        train_data = wrangle_data_GenPlus(train_ds, 'train', bs=GLOBAL_BATCH_SIZE)
        valid_data = wrangle_data_GenPlus(valid_ds, 'valid', bs=GLOBAL_BATCH_SIZE)
        test_data = wrangle_data_GenPlus(test_ds, 'test', bs=GLOBAL_BATCH_SIZE)

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

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])

    # fit the model
    history = model.fit(train_data, validation_data=valid_data, epochs=epochs)
