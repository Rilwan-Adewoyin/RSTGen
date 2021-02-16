# import matplotlib
# matplotlib.use('Agg')

import torchdata as td

#from torchdata.dataset import WrapDataset
#from torchdata.cachers import Pickle

import os
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
#os.environ['NCCL_SOCKET_IFNAME'] =  'lo' #'enp3s0'

import io

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np

import sklearn
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support 

from pathlib import Path
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import pandas as pd
import json
from functools import lru_cache

from typing import List

import pickle

from itertools import chain

from itertools import cycle, islice
from torch.utils.data._utils.collate import default_convert, default_collate

from transformers import BertTokenizer, BertModel
from transformers import get_cosine_schedule_with_warmup
import transformers

import pytorch_lightning as pl

from sklearn import preprocessing as sklp
import seaborn as sns

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import utils
import random 

import gc
from pytorch_lightning import loggers as pl_loggers

from collections import OrderedDict
import spacy
import ast
from pytorch_lightning.metrics import Metric
from multiprocessing import Pool

from typing import Optional, Any, Callable

import shutil
import tempfile
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback


from pathlib import Path

#SGNN imports
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

#Monkey Patch for logging class
def monkey_log_dir(self):
    """
        The directory for this run's tensorboard checkpoint. By default, it is named
        ``'version_${self.version}/logs'`` but it can be overridden by passing a string value
        for the constructor's version parameter instead of ``None`` or an int.
    """
    # create a pseudo standard path ala test-tube
    version = self.version if isinstance(self.version, str) else f"version_{self.version}"
    log_dir = os.path.join(self.root_dir, version,"logs")
    return log_dir

pl_loggers.TensorBoardLogger.log_dir = property( monkey_log_dir )

#suggest this change to pytorch lightning
class bAccuracy_macro(Metric):
    r"""
        Computes `balanced Accuracy <https://en.wikipedia.org/wiki/Precision_and_recall>`_:

        .. math:: \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y_i})

        Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.  Works with binary, multiclass, and multilabel
        data.  Accepts logits from a model output or integer class values in
        prediction.  Works with multi-dimensional preds and target.

        Forward accepts

        - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
        - ``target`` (long tensor): ``(N, ...)``

        If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
        This is the case for binary and multi-label logits.

        If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

        Args:
            threshold:
                Threshold value for binary or multi-label logits. default: 0.5
            compute_on_step:
                Forward only calls ``update()`` and return None if this is set to False. default: True
            dist_sync_on_step:
                Synchronize metric state across processes at each ``forward()``
                before returning the value at the step. default: False
            process_group:
                Specify the process group on which synchronization is called. default: None (which selects the entire world)
            dist_sync_fn:
                Callback that performs the allgather operation on the metric state. When `None`, DDP
                will be used to perform the allgather. default: None

        Example:
        TODO: change

            >>> from pytorch_lightning.metrics import Accuracy
            >>> target = torch.tensor([0, 1, 2, 3])
            >>> preds = torch.tensor([0, 2, 1, 3])
            >>> accuracy = Accuracy()
            >>> accuracy(preds, target)
            tensor(0.5000)
    """
    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct_positive", default=torch.zeros([num_classes]), dist_reduce_fx="sum")
        self.add_state("correct_negative", default=torch.zeros([num_classes]), dist_reduce_fx="sum")

        self.add_state("positive_samples", default=torch.zeros([num_classes]), dist_reduce_fx="sum")
        self.add_state("negative_samples", default=torch.zeros([num_classes]), dist_reduce_fx="sum")

        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        
        preds, target = pl.metrics.utils._input_format_classification(preds, target, self.threshold)
        assert preds.shape == target.shape

        correct = preds == target #(n, c)
        pos_mask= target.eq(1.0) # mask for positive samples

        self.correct_positive += torch.sum( torch.logical_and( correct, pos_mask), dim=[0] )
        self.correct_negative += torch.sum( torch.logical_and( correct, ~pos_mask), dim=[0] )

        self.positive_samples += torch.sum( pos_mask, dim=[0])
        self.negative_samples += torch.sum( ~pos_mask, dim=[0])
        
    def compute(self):
        """
        Computes macro Baccuracy over state.
        """

        pos_avg = self.correct_positive.float() / self.positive_samples
        neg_avg = self.correct_negative.float() / self.negative_samples

        vals = (pos_avg + neg_avg )/2
        
        vals = torch.masked_select( vals, ~torch.isnan(vals) )
        
        return torch.mean(vals)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        return input_ * torch.tanh(F.softplus(input_))

class DaNet(nn.Module):
    """Transformer Based Model for DA classfication Task
    """
    def __init__(self, freeze_transformer=False, dropout=0.1, 
        base_model_name='bert-base-cased', model_name="DaNet", 
        bert_output =  "PooledOutput", 
        **kwargs):

        super(DaNet, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        #D_in, H, D_out = 768, 50, 12 #transformer frozer
        
        # Instantiate BERT model
        self.base_model_name = base_model_name       
        self.model_name = model_name
        dict_transformertokenizer = utils.load_pretrained_transformer(self.base_model_name , transformer=True, tokenizer=True)
        self.transformer = dict_transformertokenizer['transformer']
        self.tokenizer = dict_transformertokenizer['tokenizer']
        self.freeze_transformer = freeze_transformer
        self.dropout = dropout
        self.bert_output = bert_output

        # Create Entity masker
        self.nem = NamedEntityMasker()

        # Freeze the Transformer model
        if self.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Create classifier Layer
        if self.bert_output == "PooledOutput":
            D_in,  D_out = 768, 17
            self.lm_head = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(D_in, D_out, bias=False ),
            )
            
            for layer in self.lm_head:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(mean=0.0, std=0.02)
                    if layer.bias is not None:
                        layer.bias.data.zero_()
    
        
        elif self.bert_output == "CLSRepresentation":
            D_in, H , D_out = 768, 96, 17

            self.lm_head = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(D_in, H, bias=False ),
                Mish(),
                nn.Dropout(self.dropout),
                nn.Linear(H, D_out, bias=False )

            )
            
            for layer in self.lm_head:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(mean=0.0, std=0.02)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        
        parser.add_argument('--config_file_m', default=None, help="Path to the \
            training hyperameters for this model ")
        
        parser.add_argument('--base_model_name', default='bert-base-cased', required=False)
        parser.add_argument('--dropout', default=0.1, required=False, help="dropout")
        parser.add_argument('--freeze_transformer', default=False, required=False, type=bool)
        parser.add_argument('--model_name', default='SGNN', required=False)
        parser.add_argument('--bert_output', default='CLSRepresentation', required=False,
                choices=['PooledOutput','CLSRepresentation'] )
        
        mparams = parser.parse_known_args( )[0]


        if mparams.config_file_m != None:
            mparams = json.load(open(utils.get_path(mparams.config_file_m)),"r" )
        
        return mparams

    def forward(self, input_):
        """[summary]

        Args:
            input_ids (torch.tensor): an input tensor with shape (batch_size,
                      max_length)

            attention_mask (torch.tensor):  a tensor that hold attention mask
                      information with shape (batch_size, max_length)

            token_type_ids (torch.tensor): an output tensor with shape (batch_size,
                      num_labels)

        Returns:
            [type]: [description]
        """

        # removing datums where the target da had no value
        input_['loss_mask'] = input_['loss_mask']>0 #creating boolean tensor

        input_['input_ids'] = input_['input_ids'][input_['loss_mask'] ]
        input_['attention_mask'] =  input_['attention_mask'][input_['loss_mask']]
        input_['token_type_ids'] = input_['token_type_ids'][input_['loss_mask']]

        #input_ids, attention_mask, token_type_ids = input_
        if input_['input_ids'].shape[0] != 1 and input_['input_ids'].dim() !=2 :
            input_ids = torch.squeeze(input_['input_ids'])
            attention_mask = torch.squeeze(input_['attention_mask'])
            token_type_ids = torch.squeeze(input_['token_type_ids'])
        else:
            input_ids = input_['input_ids']
            attention_mask = input_['attention_mask']
            token_type_ids = input_['token_type_ids']

        # Feed input to BERT
        outputs = self.transformer(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        
                
        # Experimenting with which output to use
        if self.bert_output == "PooledOutput":
            # Using Pooled Output 
            pooled_output = outputs[1]
            lm_output = pooled_output


        elif self.bert_output == "CLSRepresentation":
            # Extract the last hidden state of the token `[CLS]` for classification task
            last_hidden_state_cls = outputs[0][:, 0, :]
            lm_output = last_hidden_state_cls 


        # Feed input to classifier to compute logits
        logits = self.lm_head(lm_output)

        return logits

    def return_params(self):
        params = {}
        param_names = ['base_model_name','freeze_transformer','model_name','dropout','bert_output' ]

        # params['base_model_name'] = self.base_model_name
        # params['freeze_transformer'] = self.freeze_transformer
        # params['model_name'] = self.model_name
        # params['dropout'] = self.dropout

        params = {k:self.__dict__[k] for k in param_names}
        return params

#region SGNN
class Transformer_RBP(BaseEstimator, RandomBinaryProjections):
    """
        Class that modifies RandomBinaryProjections to use as an sklearn transformer
    """
    rand_seed = None  # Declare it as class variable
    def __init__(self, hash_name='hasher', projection_count=1, rand_seed=None):
        RandomBinaryProjections.__init__(self, hash_name, projection_count, rand_seed=rand_seed)

    def fit(self, X, y):
        self.rand = np.random.RandomState(self.rand_seed)  # rand seed after param setting
        self.reset(X.shape[1])

    def transform(self, X):
        return self.hash_vector(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def hash_vector(self, v, querying=False):
        """
        Hashes the vector and returns the binary bucket key as string.
        """
        if sp.issparse(v):
            # If vector is sparse, make sure we have the CSR representation
            # of the projection matrix
            if self.normals_csr is None:
                self.normals_csr = sp.csr_matrix(self.normals)
            # Make sure that we are using CSR format for multiplication
            if not sp.isspmatrix_csr(v):
                v = sp.csr_matrix(v)
            # Project vector onto all hyperplane normals
            # projection = self.normals_csr.dot(v)
            projection = v.dot(sp.csr_matrix.transpose(self.normals_csr))
        else:
            # Project vector onto all hyperplane normals
            projection = np.dot(v, np.matrix.transpose(self.normals))
        # Return binary key
        return projection > 0

class SGNN_tokenizer():
    
    def __init__(self, T=80, d=14):

        #build input layer
        self.pipeline = self.build_pipeline( T, d)
        
    def build_pipeline(self, T, d ):
        """
        Transformer to build the input layer, in SGNN style.
        Uses nltk skipgrams, and several Transformer_RBP layers as elements of the SGNN pipeline.
        """
        # T=80 projections for each of dimension d=14: 80 * 14 = 1120-dimensionnal word projections

        # Wrap skipgrams to use with CountVectorizer
        skipper = functools.partial(skipgrams, n=3, k=2) # 2-skip-3-grams
        # CountVectorizer params
        char_term_frequency_params = {
            'char_term_frequency__analyzer': skipper,
            'char_term_frequency__min_df': 3,
            'char_term_frequency__max_df': 0.9,
            'char_term_frequency__max_features': int(1e7),
        }

        # Transformer_RBP params
        rand_seeds = [rand.randint(0, T*100) for i in range(T)] # Need a different repeatable seed for each hasher
        hashing_feature_union_params = {
            **{'union__random_binary_projection_hasher_{}__projection_count'.format(t): d
            for t in range(T)
            },
            **{'union__random_binary_projection_hasher_{}__hash_name'.format(t): 'hasher' + str(t)
            for t in range(T)
            },
            **{'union__random_binary_projection_hasher_{}__rand_seed'.format(t): rand_seeds[t]  # only AFTER hashing.
            for t in range(T)
            }
        }

        # Pipeline to create input layer
        preprocessor = Pipeline([
            ("char_term_frequency", CountVectorizer()),
            
            ('union', FeatureUnion([
                ('random_binary_projection_hasher_{}'.format(t), Transformer_RBP())
                for t in range(T)
            ]))
            ],
        )

        params = dict()
        params.update(char_term_frequency_params)
        params.update(hashing_feature_union_params)
        preprocessor.set_params(**params)

        return preprocessor

    def fit(self, train_utterances):
        """[summary]
            Args:
                data (Pandas dataframe): A Pandas series containing a sequenceof utterances.
        """
        self.pipeline.fit( train_utterances ) #Fit on the text only
        

    # Patching Pipeline class so it can be used in a similar manner to the Huggingface tokenizers
    def __call__(self, *args_str, **kwargs ):
        """ 
            A list of current string and previous string
        """
        #TODO: consider adding method to judge which speaker is speaking
        
        encoded_input = {}
        encoded_text = self.pipeline.transform( args_str )
        encoded_input['input_hashed'] = Torch.FloatTensor(encoded_text)

        return encoded_input

class SGNN(nn.Module):

    def __init__(self, 
        T=80,
        d=14,
        dropout=0.25,
        output_type = "unilabel",
        model_name="SGNN",
        **kwargs):
        super(SGNN, self).__init__()

        self.output_type = output_type
        self.dropout = dropout 
        self.tokenizer = SGNN_tokenizer(T, d)
        self.nem = NamedEntityMasker()
        self.model_name = model_name

        self.neural_network = nn.Sequential(
                nn.Linear( 1120 , 256),
                Mish(), #Test using MISH
                nn.Dropout(self.dropout),

                nn.Linear( 256 , 256 ),
                Mish(),
                nn.Dropout(self.dropout),

                nn.Linear( 256, 17 ) )
        
        # self.neural_network2 = nn.Sequential(
        #         nn.LSTM( batch_true=False,
        #                     input_size=1120
        #                     hidden_size =256,
        #                     bidirectional=True, dropout=self.dropout),
        #         nn.Linear( 256, 17 ) )
            
        # )

        if self.output_type == "multilabel":
            #self.neural_network.add_module( 'output_activation') # = nn.Sigmoid()
            pass

        elif self.output_type == "unilabel":
            self.neural_network.add_module( 'output_activation', nn.Softmax(dim=-1) )
            
    def forward(self, input_):
        input_['loss_mask'] = input_['loss_mask']>0 #creating boolean tensor

        input_['input_hashed'] = input_['input_hashed'][input_['loss_mask'] ]

        output = self.neural_network( input_['input_hashed'] )

        return output

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        
        parser.add_argument('--config_file_m', default=None, help="Path to the \
            training hyperameters for this model ")
        
        parser.add_argument('--dropout', default=0.25, required=False, help="dropout")
        parser.add_argument('--model_name', default='SGNN', required=False)
        parser.add_argument('--T', default=80, required=True, type=int)
        parser.add_argument('--d', default=14, required=True, type=int)
        parser.add_argument('--tokenizer_path', default='./models/SGNN_tokenizer.pkl', required=False)

        
        mparams = parser.parse_known_args( )[0]

        if mparams.config_file_m != None:
            mparams = json.load(open(utils.get_path(mparams.config_file_m)),"r" )
        
        return mparams

    def return_params(self):
        #Important params for training
        list_of_params = ['batch_size', 'accumulate_grad_batches', 'learning_rate'
                'max_epochs', 'context_history_len', 
                'warmup_proportion','loss_class_weight','loss_pos_weight','ordered_label_list',
                'sub_version','tag']
        
        params = { k:v for k,v in self.__dict__.items() if k in list_of_params }

        return params
#endregion

class NamedEntityMasker():

    def __init__(self,
             batch_size=2,
             n_proc=1):

        
        self.batch_size = batch_size
        self.n_proc = n_proc
        self.nlp = spacy.load('en_core_web_sm',disable=["parser"])
        #self.nlp = en_core_web_sm.load()
        self.nlp.add_pipe(self.mask_pipe, name="Entity_mask", last=True)
        #self.nlp.add_pipe("Entity_mask", name="Entity_mask", last=True)
        
    def __call__(self,li_txt):
        return self.pipeline(li_txt)
        
    def pipeline(self,li_txt):
        return self.nlp.pipe(li_txt, as_tuples=False, batch_size = self.batch_size)

    #@Language.component('Entity_mask')
    def mask_pipe(self, doc):
        text = ''.join([token.text_with_ws if not token.ent_type else token.pos_+token.whitespace_ for token in doc])
        return text

class TrainingModule(pl.LightningModule):

    def __init__(self, batch_size=30, 
                    dir_data=None, 
                    accumulate_grad_batches=1,
                    max_epochs=80,
                    gpus=1, 
                    context_history_len=1,
                    learning_rate=8e-4,
                    warmup_proportion=0.10,
                    workers=8,
                    lr_schedule='LROnPlateau',
                    mode = 'train_new',
                    loss_class_weight = [0.08357422207100744, 0.18385707428187334, 0.32489418776359946, 0.28270948437883026, 0.29028076340137576, 0.4732623391826486, 0.35614498126635913, 0.4461250027511672, 1.8474137400874857, 0.3309135146499394, 0.38481047800771817, 2.1815475734200995, 3.423830830287625, 3.5166546883532006, 0.2784219377456878, 0.14890703368264488, 2.446652148668739],
                    loss_pos_weight = [17]*17,
                    max_length = 512,
                    mask_utterance=True,
                    tag='',
                    model=None,
                    subversion=0,
                    *args,
                    **kwargs):
        
        super(TrainingModule, self).__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        self.context_history_len = context_history_len
        self.max_length = max_length
        self.mask_utterance = mask_utterance
        
        self.loss_class_weight =  loss_class_weight
        self.loss_pos_weight = loss_pos_weight
        
           
        self.mode = mode
        self.workers = workers

        self.tag = tag
        self.subversion = subversion

        self.ordered_label_list = kwargs.get('ordered_label_list', json.load(open(utils.get_path("./label_mapping.json"),"r"))['MCONV']['labels_list']  )
        
        self.loss = nn.BCEWithLogitsLoss( weight=torch.FloatTensor( self.loss_class_weight ), 
                     pos_weight= torch.FloatTensor(self.loss_pos_weight))

        # Creating Model
        if model == None:
            mparams = kwargs.get('mparams')
            mn = mparams.get('model_name')
            
            if mn == "DaNet":
                self.model = DaNet(**mparams)
            
            elif mn == "SGNN":
                self.model = SGNN(**mparams)
                
        else:
            self.model = model


        if self.mode in ['train_new','train_cont','test','hypertune']:
            self.dir_data = utils.get_path(dir_data)
            self.max_epochs = max_epochs
            self.warmup_proportion = warmup_proportion
            self.create_data_loaders(**kwargs)

            self.learning_rate = learning_rate
            self.accumulate_grad_batches = accumulate_grad_batches

            self.step_names = ["train_label",
                    "val_label","test_label"] 
                        
            self.dict_bacc_macro =  torch.nn.ModuleDict( {
                "train_label":bAccuracy_macro(compute_on_step=True, num_classes = 17),
                "val_label" :bAccuracy_macro(compute_on_step=False,num_classes = 17),
                "test_label":bAccuracy_macro(compute_on_step=False,num_classes = 17)
            })

            self.dict_prec = torch.nn.ModuleDict( {
                k:pl.metrics.classification.Precision( multilabel=True, num_classes=17,
                    threshold=0.5,  average='macro'
                    ) for k in self.step_names } )

            self.dict_recall =  torch.nn.ModuleDict({
                k:pl.metrics.classification.Recall( multilabel=True, num_classes=17,
                    threshold=0.5, average='macro'
                    ) for k in self.step_names } )

        if self.mode in ['train_new',"hypertune"] and self.model.model_name == "SGNN" :
            self.model.tokenizer.fit( self.dg.whole_dataset("train").da )
                        
            fp = kwargs.get('mparams').get('tokenizer_path', './models/SGNN_tokenizer.pkl')
            #region: remove after debgug
            if os.path.isfile(fp):
                with open(fp,"rb") as f:
                    self.model.tokenizer = pickle.load(f)
            else:
                with open(fp, "wb" ) as f:
                    pickle.dump( self.model.tokenizer, f )
        
        elif self.mode in ["train_cont","test","inference"] and kwargs.get('model_name') == "SGNN":
            fp = kwargs.get('mparams').get('tokenizer_path')
            with open(fp,"rb") as f:
                self.model.tokenizer = pickle.load(f)
        
        # Saving training params and model params
        if self.mode in ['train_new']:
            mparams = argparse.Namespace(**self.model.return_params())
            tparams = argparse.Namespace(**self.return_params())

            utils.save_version_params(tparams, mparams, kwargs.get('version_name'), self.subversion ) 

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True,allow_abbrev=False)
  
        parser.add_argument('--dir_data', default="./combined_data", help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models")
        parser.add_argument('--max_epochs', default=80, type=int)
        parser.add_argument('-agb','--accumulate_grad_batches', default="{}", type=str)
        parser.add_argument('--context_history_len', default=1, type=int)
        parser.add_argument('-bs','--batch_size', default=32, type=int)
        parser.add_argument('-lr','--learning_rate', default=1e-3, type=float)
        parser.add_argument('-wp','--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--workers', default=8, type=int)
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','test','train_cont','hypertune'])
        parser.add_argument('--version_name', default='DaNet_v2', required=True)
        parser.add_argument('-lcw','--loss_class_weight',default="[]", required=False, type=str)
        parser.add_argument('-lpw','--loss_pos_weight',default="[]", required=False, type=str)
        parser.add_argument('--default_root_dir', default=utils.get_path("./models/") )
        parser.add_argument('-ml','--max_length', default=None, type=int)
        parser.add_argument('-sv','--subversion', default=None,required=False, type=int, help="The Experimental sub Versioning for this run" )
        parser.add_argument('--tag', default='', type=str)
        parser.add_argument('--mask_utterance',default=True, type=bool)
        parser.add_argument('--cache', default=False, type=bool)
        parser.add_argument('--model_name',required=True)
        tparams = parser.parse_known_args()[0]

        tparams.loss_class_weight = json.loads( tparams.loss_class_weight )
        tparams.loss_pos_weight = json.loads( tparams.loss_pos_weight )

        try:
            tparams.accumulate_grad_batches = int(tparams.accumulate_grad_batches)
        
        except ValueError as e:
            tparams.accumulate_grad_batches = ast.literal_eval(tparams.accumulate_grad_batches)

        return tparams

    def step(self, batch, step_name):
        target = batch.pop('da')
       
        input_= batch
        output = self.forward(input_)

        #removing lines with no DA label
        keep_mask = torch.sum(target, dim=1, dtype=bool )
        target = target[keep_mask]
        output = output[keep_mask]

        if step_name == "train":
            str_loss_key = "loss"
        else:
            str_loss_key = step_name +"_loss"

        loss = self.loss( output, target )

        _dict = { str_loss_key: loss,
            'output':torch.sigmoid(output),
            'target':target }

        return  _dict
        
    #@auto_move_data
    def forward(self, input_, *args):
        return self.model(input_)

    def format_preds(self, preds, logits=True):
        """Converts list of logit scores for Dialogue Acts
            to a list of OrderedDictionaries where
            each dict contains the DA names and probabilities 

            #done not operateon batches of predictions

        Args:
            preds ([type]): [pass in logit scores]

        Returns:
            [type]: [description]
        """
        if logits == True:
            preds = torch.sigmoid(preds)
        
        li_da = preds.tolist()

        li_dict_da = [ OrderedDict( zip(self.ordered_label_list, da) ) for da in li_da ]

        return li_da, li_dict_da

    def training_step(self, batch, batch_idx):
        output = self.step(batch,"train")
        return output

    def validation_step(self, batch, batch_idx):
        output = self.step(batch, "val")
        return output

    def test_step(self, batch, batch_idx):
        output = self.step(batch, "test")
        return output

    def training_step_end(self, outputs:dict):
        self.step_end_log( outputs, "train" )
        return outputs

    def validation_step_end(self, outputs:dict):
        self.step_end_log( outputs, "val" )        
        return outputs

    def test_step_end(self, outputs:dict):
        self.step_end_log( outputs,"test" )   
        return outputs
    
    def step_end_log(self, outputs, step_name ):

        output =  outputs['output'] 
        target  = outputs['target']
      
        if step_name == 'train':
            on_step = True
            on_epoch = False
            prog_bar = True
            logger = False
        
            # Logging train values with correct naming system
            train_loss = outputs['loss']
            train_bacc= self.dict_bacc_macro[step_name+"_label"]( output, target )
            train_prec = self.dict_recall[step_name+"_label"](output, target)
            train_rec = self.dict_prec[step_name+"_label"](output, target)   
            
            scalar_dict = {'train/loss':train_loss,
                        'train/bacc_macro':train_bacc,
                        'train/prec_macro':train_prec,
                        'train/rec_macro':train_rec}
            
            self.logger.log_metrics( scalar_dict, step=self.global_step)
            

        else:
            self.dict_bacc_macro[step_name+"_label"]( output, target )
            self.dict_recall[step_name+"_label"](output, target)
            self.dict_prec[step_name+"_label"](output, target)   
            
    def training_epoch_end(self, outputs):
        self.epoch_end_log(outputs,"train")

    def validation_epoch_end(self, outputs: List[dict]):
        self.epoch_end_log(outputs, "val")
         
    def test_epoch_end(self, outputs: List[dict]):
        self.epoch_end_log(outputs, "test")

    def epoch_end_log(self, outputs, step_name):

        if step_name == "train":
            pass
            
        else:
            #Loggin the loss
            loss = torch.stack([x[f"{step_name}_loss"] for x in outputs]).mean()
            self.log(f"{step_name}/loss", loss, logger=True, prog_bar=True)

            #Logging Aggregated Class Performance
            val_bacc = self.dict_bacc_macro[step_name+"_label"].compute()
            val_rec = self.dict_recall[step_name+"_label"].compute()
            val_prec = self.dict_prec[step_name+"_label"].compute()

            scalar_dict = {
                        'val/bacc_macro':val_bacc,
                        'val/prec_macro':val_prec,
                        'val/rec_macro':val_rec,
                        'val/f1_macro': 2*(val_prec*val_rec)/(val_prec+val_rec)  }  

            self.logger.log_metrics( scalar_dict, step=self.current_epoch)
            self.log(f"{step_name}/f1_macro",  2*(val_prec*val_rec)/(val_prec+val_rec), logger=False, prog_bar=True)
            self.log(f"{step_name}/bacc_macro",  val_bacc, logger=False, prog_bar=True)
            
            # Logging Dis-aggregated Class Performance
            class_preds = torch.cat([x["output"] for x in outputs], axis=0)     # (n,17)
            class_obsrvd = torch.cat([x["target"] for x in outputs], axis=0)   # (n, 17)
            
            class_preds = torch.where( class_preds>=0.5, 1.0 , 0.0 )
            class_preds = class_preds.detach().to('cpu').numpy().astype(np.int32)
            class_obsrvd = class_obsrvd.detach().to('cpu').numpy().astype(np.int32)

                # Creating multi-label confusion matrix image
            if self.mode != "hypertune":
                mcm = multilabel_confusion_matrix(class_obsrvd, class_preds, labels=np.arange(17) )
                li_cm = np.split(mcm, mcm.shape[0], axis=0)
                mcm_figure = self.ml_confusion_matrix( li_cm, labels=self.ordered_label_list)
                self.logger.experiment.add_figure(f"{step_name}/Confusion Matrix", mcm_figure, 
                    global_step=self.current_epoch)
            
                # Logging Precision, recall, fscore, support
            precision, recall, fscore, support = precision_recall_fscore_support(class_obsrvd, class_preds,
                labels=np.arange(17), average=None )

            tag_scalar_dict_prec2 = {f"{step_name}_precision/{k}":v for k,v in zip( self.ordered_label_list, precision )  }
            tag_scalar_dict_rec2 = {f"{step_name}_recall/{k}":v for k,v in zip( self.ordered_label_list, recall )  }
            tag_scalar_dict_f12 = {f"{step_name}_f1/{k}":v for k,v in zip( self.ordered_label_list, fscore )  }
                        
            self.logger.log_metrics( tag_scalar_dict_prec2, step = self.current_epoch )
            self.logger.log_metrics( tag_scalar_dict_rec2, step = self.current_epoch )
            self.logger.log_metrics( tag_scalar_dict_f12, step = self.current_epoch )

    def ml_confusion_matrix(self, li_cfs_matrices, labels):

        fig, ax = plt.subplots(4, 5, figsize=(12, 9), frameon=False)
         
        for axes, cfs_matrix, label in zip(ax.flatten(), li_cfs_matrices, labels):
            self.print_confusion_matrix(cfs_matrix, axes, label, ["Y", "N"])

        fig.suptitle('Multi-Label Confusion Matrix for Dialogue Acts')    
        fig.tight_layout()
        #plt.show()   

        return fig   

    def print_confusion_matrix(self, confusion_matrix, axes, class_label, class_names, fontsize=14):
        if confusion_matrix.ndim ==3:
            confusion_matrix = confusion_matrix[0]

        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,
        )

        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_xlabel('True label')
        axes.set_ylabel('Predicted label')
        axes.set_title(class_label)
        
    def create_data_loaders(self, **kwargs):
        dir_train_set = os.path.join(self.dir_data,"train") #"./combined_data/train/"
        dir_val_set = os.path.join(self.dir_data,"val")
        dir_test_set = os.path.join(self.dir_data,"test") 
        
        self.dg = DataLoaderGenerator(dir_train_set, dir_val_set,
            dir_test_set, self.batch_size, self.model.tokenizer, self.model.nem,
            workers=self.workers, mode=self.mode, max_length=self.max_length,
            cache=kwargs.get('cache',False), 
            context_history_len = self.context_history_len,
            mask_utterance = self.mask_utterance
            )
        
        self.train_dl, self.val_dl, self.test_dl = self.dg()
    
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    #@lru_cache()
    def total_steps(self):
         
        print(len(self.train_dl))

        train_batches = len( self.train_dl ) // self.gpus 
        
        if type(self.accumulate_grad_batches) == dict:

            # Caclulate the total number of steps given a accumulate_grad_batches which varies
            # -agb "{1:400, 3:200, 7:100, 11:75, 15:50, 19:40, 25:12, 29:10 }"
            df = pd.DataFrame.from_dict( self.accumulate_grad_batches, orient='index', columns=['agb_size'] )
            df.index = df.index.set_names(['start_epoch'])
            df = df.sort_index()
            df = df.reset_index()

            df['epoch_len'] = df['start_epoch'].diff(-1)

            df.loc[ len(df.index)-1 ,['epoch_len']] = self.max_epochs - df['start_epoch'].iloc[-1]

            li_epochlen_agb = df[['epoch_len','agb_size']].to_numpy().tolist()
            

            steps = sum( abs(epochs)*train_batches/agb for epochs, agb in li_epochlen_agb )

        elif type(self.accumulate_grad_batches) == int:
            
            agb = self.accumulate_grad_batches
            steps = self.max_epochs * (train_batches //  agb)

        return steps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = self.total_steps()
        warmup_steps = int( total_steps * self.warmup_proportion )

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles = 0.5
            )
        interval = "step"

        return [optimizer], [{"scheduler": lr_scheduler, "interval": interval, "monitor":"val/loss"}]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        #items.pop("v_num", None)
        return items

    def return_params(self):
        #Important params for training
        list_of_params = ['batch_size', 'accumulate_grad_batches', 'learning_rate'
                'max_epochs', 'context_history_len', 'mask_utterance',
                'warmup_proportion','loss_class_weight','loss_pos_weight','ordered_label_list',
                'max_length','sub_version','tag']
        
        params = { k:v for k,v in self.__dict__.items() if k in list_of_params }

        return params

class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """
    def __init__(self, dir_train_set, dir_val_set,
                    dir_test_set, batch_size,
                    tokenizer, nem=None,
                    context_history_len=1,
                    workers=6, mode='train_new',
                    max_length = 512, cache=False,
                    mask_utterance = True,
                    **kwargs):

        self.dir_train_set = dir_train_set
        self.dir_val_set = dir_val_set
        self.dir_test_set = dir_test_set

        self.tokenizer = tokenizer
        self.nem = nem
        label_mapping = json.load(open(utils.get_path("./label_mapping.json"),"r"))     
        self.target_binarizer = sklp.MultiLabelBinarizer()
        self.target_binarizer.fit( [label_mapping['MCONV']['labels_list'] ] )

        self.bs = batch_size
        self.context_history_len = context_history_len
        self.workers  = workers
        self.mode = mode
        self.max_length = max_length
        self.mask_utterance = mask_utterance
        self.cache = cache

    def prepare_datasets(self):
        """prepares a train, validation and test set

        Returns:
            [type]: [description]
        """
        dir_sets = [self.dir_train_set, self.dir_val_set, self.dir_test_set]
        set_names = ["train","val","test"]
        li_shuffle = [True, False, False]
        dataloaders = []
        dataloaders = [self.prepare_dataset(_dir, shuffle,name) for _dir,shuffle,name in zip(dir_sets,li_shuffle, set_names)]
        return dataloaders

    def prepare_dataset(self, dir_dset, shuffle=False, name="train"):
        """Prepares a dataloader given a directory of text files each containing one conversation

        Args:
            dir_dset ([type]): [description]
        """
        files = glob.glob( os.path.join(dir_dset,"*") )
        random.shuffle(files)

        if self.mode == "hypertune":
            li_dsets = [ SingleDataset(_f, self.tokenizer, self.nem, self.target_binarizer, 
                self.context_history_len, self.max_length,self.mask_utterance) for _f in files ]
        
        elif self.mode in ['train_new','train_cont',"test"]:
            with Pool(self.workers) as pool:
                res = pool.starmap( SingleDataset, [ (_f, self.tokenizer, self.nem, 
                        self.target_binarizer, self.context_history_len, 
                        self.max_length, self.mask_utterance) for _f in files ] )

            li_dsets = res

        if self.cache == True:
            concat_dset = td.datasets.ChainDataset(li_dsets).cache(td.cachers.Pickle(Path(f"./cachedata/{dir_dset.split('/')[-2]}_{dir_dset.split('/')[-1] }")))
        else:
            concat_dset = torch.utils.data.ConcatDataset(li_dsets)

        dataloader = torch.utils.data.DataLoader(concat_dset, batch_size=self.bs,
            shuffle=shuffle, num_workers=self.workers, collate_fn=default_collate,
             pin_memory=True )
        
        return dataloader

    def __call__(self):
        train_dl, val_dl, test_dl = self.prepare_datasets()
        return train_dl, val_dl, test_dl

    def whole_dataset(self, dset_name="train" ):
        """This functions returns the whole dataset as a pandas DataFrame
            It also removes lines that do not include dialogue label acts

        Args:
            dset_name (str, optional): [description]. Defaults to "train".
        """
        if dset_name == 'train':
            path = self.dir_train_set
        
        elif dset_name == "val":
            path = self.dir_val_set
        
        elif dset_name == "test":
            path = self.dir_test_set

        all_files = glob.glob(path + "/*.txt")

        li = []

        for filename in all_files:
            try:
                df = pd.read_csv(filename, index_col=None, header=0, sep='|')
                li.append(df)
            except pd.errors.ParserError:
                pass

        df = pd.concat(li, axis=0, ignore_index=True)
        
        # removing empty lines and non string utterances
        df = df[ ~(df.da == ' ')]
        df = df[ [isinstance(x,str) for x in df.da] ]

        return df

class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

    """
    def __init__(self, file_path, tokenizer, named_entity_masker,
            target_binarizer, context_history_len, max_length=512,
            mask_utterance=True  ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.nem = named_entity_masker
        self.target_binarizer = target_binarizer
        self.context_history_len = context_history_len
        self.max_length = max_length
        self.mask_utterance = mask_utterance

    #def parse_file(self, file_path):
        with open(self.fp, 'r') as f:
            self.data = pd.read_csv(file_path, sep='|', header=0)
            self.lines = len(self.data)
            self.data = self.data.T.values.tolist()
            _, self.li_utterances, self.li_das = self.data
                
    def __len__(self):
        return self.lines - self.context_history_len
    
    def __getitem__(self, index):

        utterances = self.li_utterances[ index:index+1+self.context_history_len ]
        utterances = [ utt if ( type(utt) == str ) else " " for utt in utterances ]

        das = self.li_das[ index+self.context_history_len ]
        
        if self.mask_utterance == True:
            masked_utterances = list(self.nem(utterances))
            encoded_input = self.encode_tokenize( masked_utterances )
        else:
            encoded_input = self.encode_tokenize( utterances )
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pre_bnrzd = das.split(" ") if ( type(das) == str ) else " "
                binarized_target = self.target_binarizer.transform( [  pre_bnrzd ] )

            #This mask highlights that the loss on this element of a batch does not have to be amaksed
            loss_mask = torch.ones((1)) 
            
        except AttributeError as e:
            binarized_target = np.zeros((1,17))
                
            #This mask highlights that the loss on this element of a batch does not have to be amaksed
            loss_mask = torch.zeros((1))

        map_datum = {**encoded_input, 'da':torch.squeeze(torch.from_numpy(binarized_target.astype(np.float))),
                        'loss_mask': loss_mask }
        return map_datum

    def encode_tokenize(self, li_str):
                
        encoded_input = self.tokenizer(*li_str, add_special_tokens=True, padding='max_length', 
            truncation=True, max_length=self.max_length, return_tensors='pt', return_token_type_ids=True )
                
        return encoded_input

def main(tparams, mparams):

    # This is the main version of the model
    # Defining callbacks
    callbacks = []

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir = utils.get_path(f'./models/'),
        name = tparams.version_name,
        version = tparams.subversion)
    
    #tb_logger.log_dir = types.MethodType( monkey_log_dir, tb_logger)    
    tparams.subversion =  tb_logger.version

    dir_model_version = os.path.join(tparams.model_dir, tparams.version_name, f"version_{tparams.subversion}")
    dir_model_version = os.path.join(tparams.model_dir, tparams.version_name, f"version_{tparams.subversion}")
    tparams.dir_checkpoints = os.path.join( dir_model_version, 'checkpoints' )
        
    checkpoint_callback = ModelCheckpoint(monitor='val/loss', save_top_k=3, 
        mode='min', dirpath=tparams.dir_checkpoints , 
        filename='{epoch:03d}-{step}')
        #filename='{epoch:03d}-{step}-{val_bAcc:03f}'
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.00,
        patience=20 ,
        verbose=False,
        mode='min')

    callbacks.append(checkpoint_callback)
    callbacks.append(early_stop_callback)    

    # Restoring model settings for continued training and testing
    if tparams.mode in ['train_cont','test']:
        checkpoint_path = utils.get_best_ckpt_path(tparams.dir_checkpoints)
        old_mparams_dict = json.load( open( os.path.join(dir_model_version,"mparam.json"),"r" ) )
        curr_mparams_dict = vars(mparams)
        old_mparams_dict.update({key: curr_tparams_dict[key] for key in ['bert_output','dropout']  })
        mparams = argparse.Namespace(** old_tparams_dict )
    
    # Restoring training settings for continued training
    if tparams.mode == "train_cont":
        # Restoring old tparams and combining with some params in new tparams
        old_tparams_dict = json.load( open( os.path.join(dir_model_version,"tparam.json"),"r" ) )
        curr_tparams_dict = vars(tparams)

        old_tparams_dict.update({key: curr_tparams_dict[key] for key in ['accumulate_grad_batches','batch_size','workers','gpus','mode','loss_pos_weight','loss_class_weight']  })
        tparams = argparse.Namespace(** old_tparams_dict )

    #danet = DaNet(**vars(mparams))    
   
    if tparams.gpus not in [0,1]:
        os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'
        os.environ['MASTER_PORT'] = '65302'
    
    # Defining Training Module and Trainer
    if tparams.mode in ["train_new"]:
        training_module = TrainingModule(mparams= vars(mparams), **vars(tparams) )

        trainer = pl.Trainer.from_argparse_args(tparams, 
                        check_val_every_n_epoch=1, logger=tb_logger,
                        default_root_dir= tparams.dir_checkpoints,
                        precision=16, callbacks=callbacks,
                        #overfit_batches = 0.1,
                        limit_train_batches = 0.1,
                        #val_check_interval = 0.5,
                        #limit_val_batches = 1,
                        #accelerator='ddp',
                        #track_grad_norm = True,
                        #,fast_dev_run=True, 
                        #log_gpu_memory=True
                        ) 

        # training_module.logger.log_hyperparams( { **training_module.return_params(),
        #                                     **training_module.model.return_params() } )

    elif tparams.mode in ["test","train_cont"]:

        accelerator = "ddp" if tparams.mode=="train_cont" else None
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        training_module = TrainingModule( mparams=vars(mparams), resume_from_checkpoint=checkpoint_path ,**vars(tparams) )
        training_module.load_state_dict(checkpoint['state_dict'])

        trainer = pl.Trainer.from_argparse_args(tparams,
                            check_val_every_n_epoch=1, logger=tb_logger,
                    default_root_dir=tparams.dir_checkpoints,
                    precision=16, callbacks=callbacks ,
                    val_check_interval = 0.5,
                    #accelerator=accelerator
                    #track_grad_norm = True,
                    #overfit_batches=5
                    #,fast_dev_run=True, 
                    #log_gpu_memory=True
                    )
                
        # load callback states
        trainer.on_load_checkpoint(checkpoint)

        # Only train_continued mode needs to load optimizer/scheduler settings
        if tparams.mode == "train_cont":
            trainer.global_step = checkpoint['global_step']
            trainer.current_epoch = checkpoint['epoch']

            # restore the optimizers
            optimizer_states = checkpoint['optimizer_states']
            
            for optimizer, opt_state in zip(trainer.optimizers, optimizer_states):
                optimizer.load_state_dict(opt_state)

                # move optimizer to GPU 1 weight at a time
                # avoids OOM
                if trainer.root_gpu is not None:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(trainer.root_gpu)

            # restore the lr schedulers
            lr_schedulers = checkpoint['lr_schedulers']
            for scheduler, lrs_state in zip(trainer.lr_schedulers, lr_schedulers):
                scheduler['scheduler'].load_state_dict(lrs_state)

        del checkpoint
        torch.cuda.empty_cache()

    if tparams.mode in ["train_new","train_cont"]:    
        trainer.fit(training_module)
        
    elif tparams.mode in ["test"]:
        training_module.eval() 
        training_module.freeze() 
        trainer.test(test_dataloaders=training_module.test_dl, model=training_module,ckpt_path=checkpoint_path)
        
def main_tune(tparams, mparams, num_samples=int(17*8), num_epochs=5):

    agb_choices = [ 80, 320]
    lcw_choices = [
            round_dp([0.008876470588235295, 0.01668235294117647, 0.026305882352941175, 0.02353529411764706, 0.024041176470588236, 0.03554705882352941, 0.028311764705882352, 0.033905882352941175, 0.10567058823529411, 0.0267, 0.030123529411764705, 0.12069999999999999, 0.1731058823529412, 0.1768470588235294, 0.023252941176470587, 0.014094117647058825, 0.1323],4), #best performing from round 1 normalised   
    ]

    lpw_choices_round_1 = [
        round_dp([ 12.29, 13.04, 14.0, 15.0 ,10.7, 11.0,  13.0, 12.0, 15.0, 10.0, 12.5, 17.0, 17.0, 17.0 , 17.0, 17.0 , 17.0],1),
        round_dp([ 6.09, 6.5, 7.0, 7.5 ,5.9, 5.5,  6.5, 6.0, 7.5, 5.0, 6.25, 8.5, 8.5, 8.5 , 8.5, 8.5 , 8.5],1),
        round_dp([ 3.04, 3.25, 3.5, 3.75 , 2.95, 2.75,  3.25, 3.0, 3.75, 2.5, 3.125, 4.25, 4.25, 4.25 , 4.25, 4.25 , 4.25],1), #best performing from round 1
        round_dp([ 3.502, 7.7042, 13.614, 11.8464, 12.1636, 19.8311, 14.9235, 18.694, 77.4121, 13.8663, 16.1247, 91.4133, 143.4686, 147.3582, 11.6667, 6.2396, 102.522],1)               
    ]
    lpw_choices = sample_vector( mu=lpw_choices_round_1[2], spread= [val/2 for val in lpw_choices_round_1[2]] , count=num_samples   )
    lpw_choices = [ round_dp(li, 2) for li in lpw_choices]

    lr_choices = [ 1e-4 ]

    config = {
    #"learning_rate": lr_choices,
    "accumulate_grad_batches":tune.choice( agb_choices),
    #"bert_output":tune.choice(["PooledOutput"]), #tune.choice(["CLSRepresentation","PooledOutput"]),
    #'dir_data':tune.choice(['./combined_data_v2'])
    "loss_class_weight":tune.choice( lcw_choices),
    "loss_pos_weight":tune.choice( lpw_choices),
    }

    scheduler = ASHAScheduler(
        max_t=tparams.get('max_epochs',7),
        grace_period=3,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "bacc", "f1","training_iteration"]) #"val/rec_macro","val/prec_macro"])

    #removing params to hypertune from tparams and mparams
    li_hypertune_params = list(config.keys())
    for k in li_hypertune_params:
        tparams.pop(k,None)
        mparams.pop(k,None)

    non_tune_params = {**tparams }

    analysis = tune.run(
        tune.with_parameters(
            tune_danet, 
                tparams=tparams,
                mparams=mparams),

        resources_per_trial={
            "cpu": tparams.get('workers'),
            "gpu": 1
        },
        metric="f1",
        mode="max",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_danet2")

    print("Best hyperparameters found were: ", analysis.best_config)

    # Saving DataFrame of results 
    results_df = analysis.results_df()
    fp = os.join( analysis._experiment_dir, "results.csv") 
    results_df.to_csv( fp, index=False)


def sample_vector(mu, spread, count=1):
    #pass in mean and spread
    #spread is calculated as a percentage of mean
    mu = np.array(mu)
    spread = np.array(spread)

    values = np.random.uniform( mu-spread, mu+spread, size=(count, len(mu) ) )

    li = values.tolist()

    return li

def tune_danet(config, tparams, mparams):
    
    if 'bert_output' not in mparams:
        mparams['bert_output'] = config['bert_output']

    module = TrainingModule( **tparams, **config , mparams=mparams)
    tblogger = TensorBoardLogger(
                save_dir=tune.get_trial_dir(), name="", version=".")

    callback_tunerreport = TuneReportCallback(
                {
                    "loss": "val/loss",
                    "f1": "val/f1_macro",
                    "bacc":"val/bacc_macro"
                },
                on="validation_end")

    # callback_tunerreport_checkpoint = TuneReportCheckpointCallback(
    #         metrics={"loss": "val/loss","bacc": "val/bacc"}
    #         filename="checkpoint",
    #         on="validation_end")

    trainer = pl.Trainer(
        gpus=tparams.get('gpus',1),
        logger=tblogger,
        progress_bar_refresh_rate=0,
        callbacks=[
            callback_tunerreport
        ],
        overfit_batches=0.05,
        checkpoint_callback=False
        )
    
    trainer.fit(module)

def round_dp(values, dp):
    li = [ round(x,dp) for x in values]
    return li

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 

    # add the trainer specific args
    tparams = TrainingModule.parse_train_specific_args(parent_parser)

    
    # add model specific args
    if tparams.model_name == "DaNet":
        mparams = DaNet.parse_model_specific_args(parent_parser)
    elif tparams.model_name == "SGNN":
        mparams = SGNN.parse_model_specific_args(parent_parser)

    if tparams.mode in ['train_new','train_cont','test','inference']:
        main(tparams, mparams)

    elif tparams.mode in ['hypertune']:
        main_tune(vars(tparams), vars(mparams))



#------------
# Training model on version 2 dataset - #ReplicationPolicy: at least one da is in the list of ones to paraphrase but "statement" never occurs
    # The class weighting used is linear

# CUDA_VISIBLE_DEVICES=0,1 python3 train.py -bs 8 --gpus 1 --max_epochs 80 --mode train_new --version_name DaNet_v002 --bert_output CLSRepresentation -sv 03 --dir_data "./combined_data_v2" -lr 1e-4 -ml 512 -lcw "[0.0836, 0.1839, 0.3249, 0.2827, 0.2903, 0.4733, 0.3561, 0.4461, 1.8474, 0.3309, 0.3848, 2.1815, 3.4238, 3.5167, 0.2784, 0.1489, 2.4467]" -lpw "[3.502, 7.7042, 13.614, 11.8464, 12.1636, 19.8311, 14.9235, 18.694, 77.4121, 13.8663, 16.1247, 91.4133, 143.4686, 147.3582, 11.6667, 6.2396, 102.522]" --tag "removed the +2 added to the positive wieght since it caused a collpase in training" -agb "{1:400, 3:200, 7:100, 11:75, 15:50, 19:40, 25:12, 29:10 }"


#------------
# Training model on version 1 dataset - #ReplicationPolicy at least one da is in the list of ones to paraphrase 
# The class weighting used is linear

# CUDA_VISIBLE_DEVICES=0,1 python3 train.py -bs 40 -agb "{1:400, 3:200, 7:100, 11:75, 15:50, 19:40, 25:12, 29:10 }" --gpus 2 --max_epochs 80 --mode train_new --version_name DaNet_v001 --dir_data "./combined_data_v2" -lr 1e-4 -ml 512 -lcw "[0.08357, 0.183, 0.3248, 0.2827, 0.290, 0.4732, 0.356, 0.44, 1.847, 0.33, 0.38, 2.18, 3.42, 3.51, 0.27, 0.148, 2.44]" -lpw "[ 12.29, 13.04, 14.0, 15.0 ,10.7, 11.0,  13.0, 12.0, 15.0, 10.0, 12.5, 17.0, 17.0, 17.0 , 17.0, 17.0 , 17.0]"

# ---- hypertuniimng

#  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --gpus 1 --batch_size 30 --version_name DaNet_v99 --mode hypertune --workers 4 --dir_data ./combined_data_v2 -ml 206 --max_epochs 5 --cache True --bert_output PooledOutput
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --gpus 1 --batch_size 30 --version_name DaNet_v99 --mode hypertune --workers 4 --dir_data ./combined_data_v2 -ml 206 --max_epochs 8 --cache True --bert_output PooledOutput -lr 1e-4