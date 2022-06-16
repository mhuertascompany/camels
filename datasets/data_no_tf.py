import numpy as np

root_path = '/net/diva/scratch-ssd1/mhuertas/users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/downloads/manual'

def generate_examples(c, l): #l being the maptype and #c the model (SIMBA or IllustrisTNG)
        fparams = root_path + '/params_'+c+'.txt'
        params = np.loadtxt(fparams)

        fmaps = root_path + "/Maps_" + l + "_"+c+"_LH_z=0.00.npy"  # he mirado los mapas en /net/diva/scratch-ssd1/mhuertas/users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/downloads/manual
        maps = np.load(fmaps)

        return maps, params