import os
import glob
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import fits
import numpy as np
import pandas as pd  # To extract the SnapNumLastMajorMerger values from tng# 100_SDSS_MajorMergers.csv

_DESCRIPTION = """
#Data of CAMELS 2D MAPS
"""

_CITATION = ""
_URL = "https://github.com/mhuertascompany/camels"


#My functions added



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
                'omega_m': tf.float32,
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

        fparams_tng = root_path + '/params_IllustrisTNG.txt'
        fparams_simba = root_path + '/params_SIMBA.txt'
        params_tng = np.loadtxt(fparams_tng)
        params_simba = np.loadtxt(fparams_simba)

        labels = ['Mgas', 'Mstar']


        for c, l in enumerate(labels):
            fmaps_tng = root_path+"/Maps_"+l+"_IllustrisTNG_LH_z=0.00.npy" #he mirado los mapas en /net/diva/scratch-ssd1/mhuertas/users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/downloads/manual
            fmaps_simba = root_path+"/Maps_"+l+"_SIMBA_LH_z=0.00.npy"
            maps_tng = np.load(fmaps_tng)
            maps_simba = np.load(fmaps_simba)
            if c == 0:
                map_dict_tng = {l: np.expand_dims(maps_tng.astype('float32'), axis=3)}
                map_dict_simba = {l: np.expand_dims(maps_simba.astype('float32'), axis=3)} #.astype: cast a pandas object to a specified dtype
            else:
                map_dict_tng.update({l: np.expand_dims(maps_tng.astype('float32'), axis=3)})
                map_dict_simba.update({l: np.expand_dims(maps_simba.astype('float32'), axis=3)})


#Para IllustrisTNG

        for i in range(len(maps_tng)):

            if True:
                # Opening images
                for c,l in enumerate(labels):
                    map_tng = map_dict_tng[l]
                    if c==0:
                        example_tng = {l: map_tng[i].astype('float32')}
                    else:
                        example_tng.update({l:map_tng[i].astype('float32')})

                params_map_tng = params_tng[i // 15]
                example_tng.update({'omega_m': params_map_tng[0]})
                example_tng.update({'sigma_8': params_map_tng[1]})
                example_tng.update({'A_sn1': params_map_tng[2]})
                example_tng.update({'A_agn1': params_map_tng[3]})
                example_tng.update({'A_sn2': params_map_tng[4]})
                example_tng.update({'A_agn2': params_map_tng[5]})



                yield i, example_tng
            else:
                continue


#Para SIMBA

        for i in range(len(maps_simba)):

            if True:

                for c, l in enumerate(labels):
                    map_simba = map_dict_simba[l]
                    if c == 0:
                        example_simba = {l: map_simba[i].astype('float32')}
                    else:
                        example_simba.update({l: map_simba[i].astype('float32')})

                params_map_simba = params_simba[i // 15]
                example_simba.update({'omega_m': params_map_simba[0]})
                example_simba.update({'sigma_8': params_map_simba[1]})
                example_simba.update({'A_sn1': params_map_simba[2]})
                example_simba.update({'A_agn1': params_map_simba[3]})
                example_simba.update({'A_sn2': params_map_simba[4]})
                example_simba.update({'A_agn2': params_map_simba[5]})



                yield i, example_simba
            else:
                continue


