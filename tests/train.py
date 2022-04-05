from datasets import maps
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

# load datasets

data_dir  = "/net/diva/scratch-ssd1/mhuertas/users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data"
#dset = tfds.load('maps', split='train', data_dir=data_dir)

def build_model(nfilters, num_components, input_shape, output_shape):
    cnn = tfk.Sequential([
        tfkl.Conv2D(
            nfilters, (4, 4),
            input_shape=(input_shape, input_shape, 1),
            padding="same",
            activation='relu'),
        tfkl.BatchNormalization(),
        tfkl.MaxPool2D((2, 2), strides=2),
        tfkl.Conv2D(
            nfilters * 2, (3, 3),
            padding="same",
            activation='relu'),
        tfkl.MaxPool2D((2, 2), strides=2),
        tfkl.Conv2D(
            nfilters * 4, (2, 2),
            padding="same",
            activation='relu'),
        tfkl.MaxPool2D((2, 2), strides=2),

        tf.keras.layers.Flatten(),
        tfkl.Dense(128, activation='relu'),
        tfkl.Dense(64, activation='relu'),
        tfkl.Dense(64, activation='tanh'),
        tfkl.Dense(tfpl.MixtureNormal.params_size(num_components), activation=None),
        tfpl.MixtureNormal(num_components)
    ])

    cnn.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00002), loss=negloglik)

    return cnn


def input_fn(mode='train', batch_size=64):
    """
    mode: 'train' or 'test'
    """
    if mode == 'train':
        dataset = tfds.load('sfhsed', split='train[:80%]',data_dir=data_dir)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)
    else:
        dataset = tfds.load('sfhsed', split='train[80%:]')

    dataset = dataset.batch(batch_size, drop_remainder=True)
    #dataset = dataset.map(preprocessing) # Apply data preprocessing
    dataset = dataset.prefetch(-1)  # fetch next batches while training current one (-1 for autotune)
    return dataset


dset = input_fn()

history = model.fit(x=dset['flux'], y=dset['quantile'][4], epochs=20)