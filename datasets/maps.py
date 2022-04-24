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

    def _info(self) -> tfds.core.DatasetInfo:
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
            fmaps_TNG = root_path+"/Maps_"+l+"_IllustrisTNG_LH_z=0.00.npy"
            fmaps_SIMBA = root_path+"/Maps_"+l+"_SIMBA_LH_z=0.00.npy"
            maps_TNG = np.load(fmaps_TNG)
            maps_SIMBA = np.load(fmaps_SIMBA)
            if c ==0:
                map_dict = {l:np.expand_dims(maps.astype('float32'),axis=3)}
            else:
                map_dict.update({l:np.expand_dims(maps.astype('float32'),axis=3)})



        for i in range(len(maps)):

            if True:
                # Opening images
                for c,l in enumerate(labels):
                    map = map_dict[l]
                    if c==0:
                        example = {l: map[i].astype('float32')}
                    else:
                        example.update({l:map[i].astype('float32')})

                params_map = params[i // 15]
                example.update({'omega_m': params_map[0]})
                example.update({'sigma_8': params_map[1]})
                example.update({'A_sn1': params_map[2]})
                example.update({'A_agn1': params_map[3]})
                example.update({'A_sn2': params_map[4]})
                example.update({'A_agn2': params_map[5]})



                yield i, example
            else:
                continue
