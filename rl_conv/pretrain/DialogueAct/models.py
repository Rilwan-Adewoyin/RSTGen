# Cell
import functools
import random as rand
import re

from nearpy.hashes import RandomBinaryProjections
from nltk.util import skipgrams
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from train.py import TrainingModule, Mish, NamedEntityMasker
