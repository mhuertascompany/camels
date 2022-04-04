ef build_model(nfilters, num_components, input_shape, output_shape):
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