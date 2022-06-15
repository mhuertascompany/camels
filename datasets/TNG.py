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


def _split_generators(self, dl):
  """Returns generators according to split"""
  return {tfds.Split.TRAIN: self._generate_examples(str(dl.manual_dir))}


def load_maps(maptype):

  fmaps = root_path+'/Maps_'+maptype+'_IllustrisTNG_LH_z=0.00.npy'
  maps  = np.load(fmaps)
  fparams = root_path +'/params_IllustrisTNG.txt'
  params  = np.loadtxt(fparams)
  return maps,params


