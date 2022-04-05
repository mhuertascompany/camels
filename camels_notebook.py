import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import tensorflow as tf
import tensorflow_probability as tfp
import sklearn

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
plt.style.use(astropy_mpl_style)

path = "/net/diva/scratch-ssd1/mhuertas/users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/downloads/manual"

negloglik = lambda y, p_y: -p_y.log_prob(y)


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


# load temperature map - This can take a while
print("Loading data...")
fmaps = path+'/Maps_T_IllustrisTNG_LH_z=0.00.npy'
maps  = np.load(fmaps)

fparams = path+'/params_IllustrisTNG.txt'
params  = np.loadtxt(fparams)

map_number = 1250
#plt.imshow(np.log10(maps[map_number]),cmap=plt.get_cmap('binary_r'), origin='lower', interpolation='bicubic')
#plt.show()

params_map = params[map_number//15]
print('Value of the parameters for this map')
print('Omega_m: %.5f'%params_map[0])
print('sigma_8: %.5f'%params_map[1])
print('A_SN1:   %.5f'%params_map[2])
print('A_AGN1:  %.5f'%params_map[3])
print('A_SN2:   %.5f'%params_map[4])
print('A_AGN2:  %.5f'%params_map[5])

ntrain = 3000
ntest = 300

Y=[]
for i in range(maps.shape[0]):
  Y.append(params[i//15][0])


X, Y = sklearn.utils.shuffle(maps, Y)

Xtrain=X[0:ntrain]
Ytrain = Y[0:ntrain]

Xtest = X[ntrain:ntrain+ntest]
Ytest = Y[ntrain:ntrain+ntest]

print("Training the model..")
model1 = build_model(16,1,256,1)
model1.fit(np.expand_dims(Xtrain,axis=3),np.array(Ytrain),epochs = 5, validation_data=(np.expand_dims(Xtest,axis=3),np.array(Ytest)))