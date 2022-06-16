import numpy as np

root_path = '/net/diva/scratch-ssd1/mhuertas/users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/downloads/manual'

def generate_examples(l): #l being the maptype
        fparams = root_path + '/params_SIMBA.txt'
        params = np.loadtxt(fparams)

        fmaps = root_path + "/Maps_" + l + "_SIMBA_LH_z=0.00.npy"  # he mirado los mapas en /net/diva/scratch-ssd1/mhuertas/users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/downloads/manual
        maps = np.load(fmaps)

        return maps, params