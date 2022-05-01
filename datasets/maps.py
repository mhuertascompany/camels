import os
import glob
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import fits
import numpy as np
import pandas as pd  # To extract the SnapNumLastMajorMerger values from TNG100_SDSS_MajorMergers.csv

_DESCRIPTION = """
#Data of CAMELS 2D MAPS
"""

_CITATION = ""
_URL = "https://github.com/mhuertascompany/camels"


## My functions added ##



#######################

class maps(tfds.core.GeneratorBasedBuilder):
    """TNG100 galaxy dataset"""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {'1.0.0': 'Initial release.', }
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Nothing to download. Dataset was generated at first call."

    def _info(self) -> tfds.core.DatasetInfo:   #https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo
        """Returns the dataset metadata."""
        N_TIMESTEPS = 100
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            homepage=_URL,
            citation=_CITATION,
            # Two features: image with 3 channels (stellar light, velocity map, velocity dispersion map)
            #  and redshift value of last major merger
            features=tfds.features.FeaturesDict({
                'Mgas': tfds.features.Tensor(shape=(256, 256, 1), dtype=tf.float32),
                'Mstar': tfds.features.Tensor(shape=(256, 256, 1), dtype=tf.float32),
                'omega_m':tf.float32,
                'sigma_8': tf.float32,
                "A_sn1": tf.float32,
                "A_agn1": tf.float32,
                "A_sn2": tf.float32,
                "A_agn2": tf.float32
            }),
            supervised_keys=('noiseless_griz', 'last_major_merger'),
        )

    def _split_generators(self, dl):
        """Returns generators according to split"""
        return {tfds.Split.TRAIN: self._generate_examples(str(dl.manual_dir))}

    def _generate_examples(self, root_path):
        """Yields examples."""

        fparams_TNG = root_path + '/params_IllustrisTNG.txt'
        fparams_SIMBA = root_path + '/params_SIMBA.txt'
        params_TNG = np.loadtxt(fparams_TNG)
        params_SIMBA = np.loadtxt(fparams_SIMBA)

        labels = ['Mgas','Mstar']


        for c,l in enumerate(labels):
            fmaps_TNG = root_path+"/Maps_"+l+"_IllustrisTNG_LH_z=0.00.npy" #he mirado los mapas en /net/diva/scratch-ssd1/mhuertas/users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/downloads/manual
            fmaps_SIMBA = root_path+"/Maps_"+l+"_SIMBA_LH_z=0.00.npy"
            maps_TNG = np.load(fmaps_TNG)
            maps_SIMBA = np.load(fmaps_SIMBA)
            if c ==0:
                map_dict_TNG = {l:np.expand_dims(maps_TNG.astype('float32'),axis=3)}
                map_dict_SIMBA = {l:np.expand_dims(maps_SIMBA.astype('float32'),axis=3)} #.astype: cast a pandas object to a specified dtype
            else:
                map_dict_TNG.update({l:np.expand_dims(maps_TNG.astype('float32'),axis=3)})
                map_dict_SIMBA.update({l: np.expand_dims(maps_SIMBA.astype('float32'), axis=3)})


#Para IllustrisTNG

        for i in range(len(maps_TNG)):

            if True:
                # Opening images
                for c,l in enumerate(labels):
                    map_TNG = map_dict_TNG[l]
                    if c==0:
                        example_TNG = {l: map_TNG[i].astype('float32')}
                    else:
                        example_TNG.update({l:map_TNG[i].astype('float32')})

                params_map_TNG = params_TNG[i // 15]
                example_TNG.update({'omega_m': params_map_TNG[0]})
                example_TNG.update({'sigma_8': params_map_TNG[1]})
                example_TNG.update({'A_sn1': params_map_TNG[2]})
                example_TNG.update({'A_agn1': params_map_TNG[3]})
                example_TNG.update({'A_sn2': params_map_TNG[4]})
                example_TNG.update({'A_agn2': params_map_TNG[5]})



                yield i, example_TNG
            else:
                continue


#Para SIMBA

        for i in range(len(maps_SIMBA)):

            if True:

                for c, l in enumerate(labels):
                    map_SIMBA = map_dict_SIMBA[l]
                    if c == 0:
                        example_SIMBA = {l: map_SIMBA[i].astype('float32')}
                    else:
                        example_SIMBA.update({l: map_SIMBA[i].astype('float32')})

                params_map_SIMBA = params_SIMBA[i // 15]
                example_SIMBA.update({'omega_m': params_map_SIMBA[0]})
                example_SIMBA.update({'sigma_8': params_map_SIMBA[1]})
                example_SIMBA.update({'A_sn1': params_map_SIMBA[2]})
                example_SIMBA.update({'A_agn1': params_map_SIMBA[3]})
                example_SIMBA.update({'A_sn2': params_map_SIMBA[4]})
                example_SIMBA.update({'A_agn2': params_map_SIMBA[5]})



                yield i, example_SIMBA
            else:
                continue

print(map_SIMBA)