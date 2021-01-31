import os
os.environ['NCCL_SOCKET_IFNAME'] =  'lo' #'enp3s0'
import io
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sklearn

from pathlib import Path

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


#from pytorch_lightning.loggers import TensorBoardLogger

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
        dict_transformertokenizer = utils.load_pretrained_transformer(self.base_model_name , transformer=True, tokenizer=True)
        self.transformer = dict_transformertokenizer['transformer']
        self.tokenizer = dict_transformertokenizer['tokenizer']
        self.freeze_transformer = freeze_transformer
        self.dropout = dropout
        self.bert_output =bert_output

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
        parser.add_argument('--model_name', default='DaNet', required=False)
        parser.add_argument('--bert_output', default='PooledOutput', required=False,
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

        params['base_model_name'] = self.base_model_name
        params['freeze_transformer'] = self.freeze_transformer
        params['model_name'] = self.base_model_name
        params['dropout'] = self.dropout
        return params

class NamedEntityMasker():

    def __init__(self,
             batch_size=2,
             n_proc=1):

        
        self.batch_size = batch_size
        self.n_proc = n_proc
        self.nlp = spacy.load('en_core_web_sm',disable=["parser"])
        self.nlp.add_pipe(self.mask_pipe, name="Entity_mask", last=True)
        
    def __call__(self,li_txt):
        return self.pipeline(li_txt)
        
    def pipeline(self,li_txt):
        return self.nlp.pipe(li_txt, as_tuples=False, batch_size = self.batch_size)

    def mask_pipe(self, document):
        text = ''.join([token.text_with_ws if not token.ent_type else token.pos_+token.whitespace_ for token in document])
        
        return text

class TrainingModule(pl.LightningModule):

    def __init__(self, model=DaNet(), batch_size=20, 
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
                    tag='',
                    subversion=0,
                    *args,
                    **kwargs):
        super(TrainingModule, self).__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        self.context_history_len = context_history_len
        self.max_length = max_length
        
        self.loss_class_weight =  loss_class_weight
        self.loss_pos_weight = loss_pos_weight
        
        self.model = model
        self.mode = mode
        self.workers = workers

        self.tag = tag
        self.subversion = subversion


        self.ordered_label_list = kwargs.get('ordered_label_list', json.load(open(utils.get_path("./label_mapping.json"),"r"))['MCONV']['labels_list']  )
        

        self.loss = nn.BCEWithLogitsLoss( weight=torch.FloatTensor( self.loss_class_weight ), 
                     pos_weight= torch.FloatTensor(self.loss_pos_weight), 
                      )
        
        if self.mode in ['train_new','train_cont','test']:
            self.dir_data = utils.get_path(dir_data)
            self.max_epochs = max_epochs
            self.warmup_proportion = warmup_proportion
            self.lr_schedule = lr_schedule
            self.create_data_loaders()
            self.learning_rate = learning_rate
            self.accumulate_grad_batches = accumulate_grad_batches

            self.step_names = ["train_label",
                    "val_label","test_label"] 
                        

            self.dict_acc_micro =  torch.nn.ModuleDict( {
                "train_label":pl.metrics.classification.Accuracy(),
                "val_label" :pl.metrics.classification.Accuracy(compute_on_step=False),
                "test_label":pl.metrics.classification.Accuracy(compute_on_step=False)
            })

            self.dict_prec = torch.nn.ModuleDict( {
                k:pl.metrics.classification.Precision( multilabel=True, num_classes=17,
                    threshold=0.5,  average='macro'
                    ) for k in self.step_names } )

            self.dict_recall =  torch.nn.ModuleDict({
                k:pl.metrics.classification.Recall( multilabel=True, num_classes=17,
                    threshold=0.5, average='macro'

                    ) for k in self.step_names } )

        # Saving training params and model params
        if self.mode in ['train_new']:
            mparams = argparse.Namespace(**model.return_params())
            tparams = argparse.Namespace(**self.return_params())

            utils.save_version_params(tparams, mparams, kwargs.get('version_name'), self.subversion ) 

            

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True,allow_abbrev=False)
  
        parser.add_argument('--dir_data', default="./combined_data", help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models")
        parser.add_argument('--max_epochs', default=80, type=int)
        parser.add_argument('-agb','--accumulate_grad_batches', default=[], type=str)
        parser.add_argument('--context_history_len', default=1, type=int)
        parser.add_argument('-bs','--batch_size', default=32, type=int)
        parser.add_argument('-lr','--learning_rate', default=1e-3, type=float)
        parser.add_argument('-wp','--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--workers', default=8, type=int)
        parser.add_argument('--gpus', default=4, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','test','train_cont'])
        parser.add_argument('--version_name', default='DANet_V000', required=True)
        parser.add_argument('--lr_schedule', default='cosine_warmup', required=False, choices =['LROnPlateau','cosine_warmup'])
        parser.add_argument('--loss_type',default="BCE", required=False, type=str)
        parser.add_argument('-lcw','--loss_class_weight',default=[], required=True, type=str)
        parser.add_argument('-lpw','--loss_pos_weight',default=[], required=True, type=str)
        parser.add_argument('--default_root_dir', default=utils.get_path("./models/") )
        parser.add_argument('-ml','--max_length', default=512, type=int)
        parser.add_argument('-sv','--subversion', default=None,required=False, type=int, help="The Experimental sub Versioning for this run" )
        parser.add_argument('--tag', default='', type=str)


        tparams = parser.parse_known_args()[0]

        tparams.loss_class_weight = json.loads( tparams.loss_class_weight )
        tparams.loss_pos_weight = json.loads( tparams.loss_pos_weight )
        #Since we are more concerned with false negatives than false positives, the lpw is increased by 1 for all classes
        #tparams.loss_pos_weight = [ val+2 for val in tparams.loss_pos_weight ]

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
            'output':output,
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
        output_bnrzd = torch.where( output<0.5,0.0,1.0)

        correct_micro = output_bnrzd.eq(target).min(dim=1)[0]

        ones =  torch.ones_like(correct_micro).type_as(correct_micro)

        self.dict_acc_micro[step_name+"_label"]( correct_micro, ones )

        self.dict_recall[step_name+"_label"](output, target)
        self.dict_prec[step_name+"_label"](output, target)   
      
        if step_name == 'train':
            on_step = True
            on_epoch = False
            prog_bar = True
            logger = False

        else:
            on_step = False
            on_epoch = True
            prog_bar = False
            logger = True        
        
        self.log(f'{step_name}_acc_micro', self.dict_acc_micro[step_name+"_label"].compute(), on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)
        self.log(f'{step_name}_rec', self.dict_recall[step_name+"_label"].compute(), on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger )
        self.log(f'{step_name}_prec', self.dict_prec[step_name+"_label"].compute(), on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)             

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
            self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True)

            #Logging extra analysis
            class_preds = torch.stack([x[f"{step_name}_class_preds"] for x in outputs])     # (n,17)
            class_obsrvd = torch.stack([x[f"{step_name}_class_obsrvd"] for x in outputs])   # (n, 17)

            # Building confusion matrix
            cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
            cm_figure = plot_confusion_matrix(cm, class_names=class_names)
            self.tb_logger.experiment.add_figure("Confusion Matrix",cm_figure, 
                    global_step=self.current_epoch)

            # Getting class specific accuracy results 
            class_accuracies = pl.metrics.functional.classification.accuracy(class_preds, class_obsrvd,
                            num_classes=17, class_reduction='none' ) #( 17)

            tag_scalar_dict = {k:v for k,v in zip( self.ordered_label_list, class_accuracies )  }
            self.tb_logger.experiment.add_scalars( 'class_accuracy', tag_scalar_dict, 
                global_step = self.current_epoch)

            # Getting class specific recall and precision and f1 
            class_f1 = pl.metrics.functional.classification.f1(class_preds, class_obsrvd,
                            num_classes=17, class_reduction='none', multilabel=True ) #( 17)





    def plot_confusion_matrix(self, cm):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        
        Args:
           cm (array, shape = [n, n]): a confusion matrix of integer classes
           class_names (array, shape = [n]): String names of the integer classes
        """
        class_names = self.ordered_label_list
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        return figure

    def plot_to_image(figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
    
        buf = io.BytesIO()
        
        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format='png')
        
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        
        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        
        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)
    
    return image

    def create_data_loaders(self, **kwargs):
        dir_train_set = os.path.join(self.dir_data,"train") #"./combined_data/train/"
        dir_val_set = os.path.join(self.dir_data,"val")
        dir_test_set = os.path.join(self.dir_data,"test") 
        
        dg = DataLoaderGenerator(dir_train_set, dir_val_set,
            dir_test_set, self.batch_size, self.model.tokenizer, self.model.nem,
            workers=self.workers, mode=self.mode, max_length=self.max_length
            )
        
        self.train_dl, self.val_dl, self.test_dl = dg()
    
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    @lru_cache()
    def total_steps(self):
         
        train_batches = len( self.train_dl ) //self.gpus 
        
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
            steps = (self.max_epochs * train_batches) //  abg

        return steps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = self.total_steps()
        warmup_steps = int( total_steps * self.warmup_proportion )

        if self.lr_schedule == "cosine_warmup" :
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles = 0.5
                )
            interval = "step"
        
        elif self.lr_schedule == "LROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,)
            interval = "epoch"

        return [optimizer], [{"scheduler": lr_scheduler, "interval": interval, "monitor":"val_loss"}]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def return_params(self):
        #Important params for training
        list_of_params = ['batch_size', 'accumulate_grad_batches', 'learning_rate'
                'max_epochs', 'context_history_len', 'learning_rate','lr_schedule',
                'warmup_proportion','loss_class_weight','loss_pos_weight','ordered_label_list',
                'max_length','sub_version','tag']
        
        params = { k:v for k,v in self.__dict__.items() if k in list_of_params }

        return params
    
class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """
    def __init__(self, dir_train_set, dir_val_set,
                    dir_test_set, batch_size,
                    tokenizer, nem,
                    context_history_len=1,
                    workers=0, mode='train_new',
                    max_length = 512,
                    **kwargs):

        self.dir_train_set = dir_train_set
        self.dir_val_set = dir_val_set
        self.dir_test_set = dir_test_set

        self.tokenizer = tokenizer
        self.nem = nem
        label_mapping = json.load(open(utils.get_path("./label_mapping.json"),"r"))     
        self.target_binarizer = sklp.MultiLabelBinarizer()
        self.target_binarizer.fit( [label_mapping['MCONV']['labels_list'] ] )

            #Add a utility code that downloads pretrained bert tokenizer if it is not already in the relatively directory above
        self.bs = batch_size
        self.context_history_len = context_history_len
        self.workers  = workers
        self.mode = mode
        self.max_length = max_length

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
        li_dsets = [ SingleDataset(_f, self.tokenizer, self.nem, self.target_binarizer, 
            self.context_history_len, self.max_length) for _f in files ]
                
        concat_dset = torch.utils.data.ConcatDataset(li_dsets)

        dataloader = torch.utils.data.DataLoader(concat_dset, batch_size=self.bs,
            shuffle=shuffle, num_workers=self.workers, collate_fn=default_collate,
             pin_memory=True )
        
        return dataloader

    def __call__(self):
        
        train_dl, val_dl, test_dl = self.prepare_datasets()
        return train_dl, val_dl, test_dl
    
class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

    """
    def __init__(self, file_path, tokenizer, named_entity_masker ,target_binarizer, context_history_len, max_length=512  ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.nem = named_entity_masker
        self.target_binarizer = target_binarizer
        self.context_history_len = context_history_len
        self.max_length = max_length

    #def parse_file(self, file_path):
        with open(self.fp, 'r') as f:
            self.data = pd.read_csv(file_path, sep='|', header=0)
            self.lines = len(self.data)
            self.data = self.data.T.values.tolist()
            _, self.li_utterances, self.li_das = self.data
            #self.li_utterances = list(self.nem(self.li_utterances))
                
    def __len__(self):
        return self.lines - self.context_history_len
    
    def __getitem__(self, index):

        
        utterances = self.li_utterances[ index:index+1+self.context_history_len ]
        utterances = [ utt if ( type(utt) == str ) else " " for utt in utterances ]

        das = self.li_das[ index+self.context_history_len ]

        masked_utterances = list(self.nem(utterances))
        encoded_input = self.encode_tokenize( masked_utterances )
        
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
        #_str_tknized = self.tokenizer.tokenize(_str)
        
        encoded_input = self.tokenizer(*li_str, add_special_tokens=True, padding='max_length', 
            truncation=True, max_length=self.max_length, return_tensors='pt', return_token_type_ids=True )
                
        return encoded_input

def main(tparams, mparams):

    # This is the main version of the model
    #model_dir = tparams.model_dir
    
    # Defining callbacks
    callbacks = []

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir = utils.get_path(f'./models/'),
        name = tparams.version_name,
        version = tparams.subversion
        )
    tparams.subversion =  tb_logger.version

    
    dir_model_version = os.path.join(tparams.model_dir, tparams.version_name, f"version_{tparams.subversion:02d}")
    tparams.dir_checkpoints = os.path.join( dir_model_version, 'checkpoints' )
        
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=3, 
        mode='min', dirpath=tparams.dir_checkpoints , 
        filename='{epoch:03d}-{val_loss:.3f}-{val_acc_micro:.3f}')
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='auto')

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

    danet = DaNet(**vars(mparams))
    

    # Defining Trainer
    if tparams.gpus not in [0,1]:
        os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'
        os.environ['MASTER_PORT'] = '65302'

    if tparams.mode in ["train_new"]:
        training_module = TrainingModule(**vars(tparams), model=danet )
        
        trainer = pl.Trainer.from_argparse_args(tparams, 
                        check_val_every_n_epoch=1, logger=tb_logger,
                        default_root_dir= tparams.dir_checkpoints,
                        precision=16, callbacks=callbacks,
                        overfit_batches = 0.1,
                        #limit_train_batches = 0.2,
                        #val_check_interval = 0.5,
                        accelerator='ddp'
                        #track_grad_norm = True,
                        #overfit_batches=5
                        #,fast_dev_run=True, 
                        #log_gpu_memory=True
                        ) 
        
    # Making training module
    elif tparams.mode in ["test","train_cont"]:


        accelerator = "ddp" if tparams.mode=="train_cont" else None

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        training_module = TrainingModule(**vars(tparams), model=danet, resume_from_checkpoint=checkpoint_path )
        training_module.load_state_dict(checkpoint['state_dict'])

        #training_module.to(device)

        trainer = pl.Trainer.from_argparse_args(tparams,
                            check_val_every_n_epoch=1, logger=tb_logger,
                    default_root_dir=tparams.dir_checkpoints,
                    precision=16, callbacks=callbacks ,
                    val_check_interval = 0.5,
                    accelerator=accelerator
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
        
if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 
    
    # add model specific args
    mparams = DaNet.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = TrainingModule.parse_train_specific_args(parent_parser)

    main(tparams, mparams)

#    CUDA_VISIBLE_DEVICES=0 python3 train.py -bs 32 -agb 560 --workers 8 --gpus 1 --max_epochs 50  --learning_rate 1e-3 --warmup_proportion 0.1

#   CUDA_VISIBLE_DEVICES=0,1,2 python3 train.py -bs 8 -agb 20 --workers 8 --gpus 1  --max_epochs 70  --learning_rate 1e-4 

# Training model on version 1 dataset - #At least one da is in the list of ones to paraphrase
# CUDA_VISIBLE_DEVICES=0 python3 train.py -bs 32 -agb 20 --workers 8 --gpus 1 --max_epochs 70  --learning_rate 1e-3 --mode train_cont --version_name DaNet_v027


#------------
# Training model on version 2 dataset - #at least one da is in the list of ones to paraphrase but "statement" never occurs
    # The class weighting used is linear

# CUDA_VISIBLE_DEVICES=0,1 python3 train.py -bs 56 -agb 10 --gpus 2 --max_epochs 80 --mode train_new --version_name DaNet_v002 --bert_output CLSRepresentation --dir_data "./combined_data_v2" -lr 1e-4 -ml 512 -lcw "[0.08357, 0.183, 0.3248, 0.2827, 0.290, 0.4732, 0.356, 0.44, 1.847, 0.33, 0.38, 2.18, 3.42, 3.51, 0.27, 0.148, 2.44]" -lpw "[ 12.29, 13.04, 14.0, 15.0 ,10.7, 11.0,  13.0, 12.0, 15.0, 10.0, 12.5, 17.0, 17.0, 17.0 , 17.0, 17.0 , 17.0]" --tag "removed the +2 added to the positive wieght since it caused a collpase in training" -agb "{1:400, 3:200, 7:100, 11:75, 15:50, 19:40, 25:12, 29:10 }" -sv 01


#------------
# Training model on version 1 dataset - #at least one da is in the list of ones to paraphrase 
# The class weighting used is linear

# CUDA_VISIBLE_DEVICES=0,1 python3 train.py -bs 40 -agb "{1:400, 3:200, 7:100, 11:75, 15:50, 19:40, 25:12, 29:10 }" --gpus 2 --max_epochs 80 --mode train_new --version_name DaNet_v002 --bert_output CLSRepresentation --dir_data "./combined_data_v2" -lr 1e-4 -ml 512 -lcw "[0.08357, 0.183, 0.3248, 0.2827, 0.290, 0.4732, 0.356, 0.44, 1.847, 0.33, 0.38, 2.18, 3.42, 3.51, 0.27, 0.148, 2.44]" -lpw "[ 12.29, 13.04, 14.0, 15.0 ,10.7, 11.0,  13.0, 12.0, 15.0, 10.0, 12.5, 17.0, 17.0, 17.0 , 17.0, 17.0 , 17.0]"


