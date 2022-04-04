from datasets import maps
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

# load dataset

data_dir  = "/net/diva/scratch-ssd1/mhuertas/users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data"
dset = tfds.load('maps', split='train', data_dir=data_dir)

