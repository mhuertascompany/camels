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


class TNG(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {'1.0.0': 'Initial release.', }
  MANUAL_DOWNLOAD_INSTRUCTIONS = "Nothing to download. Dataset was generated at first call."

  def _info(self) -> tfds.core.DatasetInfo:
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

  def load_maps(maptype):
    fmaps = root_path+'/Maps_'+maptype+'_IllustrisTNG_LH_z=0.00.npy'
    maps  = np.load(fmaps)
    fparams = root_path +'/params_IllustrisTNG.txt'
    params  = np.loadtxt(fparams)
    return maps,params


