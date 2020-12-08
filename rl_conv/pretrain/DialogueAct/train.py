import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import glob
import pandas as pd
import json
from functools import lru_cache

from typing import List




from itertools import cycle, islice
from torch.utils.data._utils.collate import default_convert, default_collate

from transformers import BertTokenizer, BertModel
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

import pytorch_lightning as pl
from sklearn import preprocessing as sklp
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import utils
import random 

import gc
from pytorch_lightning import loggers as pl_loggers

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        return input_ * torch.tanh(F.softplus(input_))

class TrainingModule(pl.LightningModule):

    def __init__(self, batch_size, accumulate_grad_batches,
                    max_epochs, gpus, dir_data,
                    model, context_history_len,
                    learning_rate,
                    warmup_proportion,
                    **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_epochs = max_epochs
        self.gpus =  gpus
        self.dir_data = utils.get_path(dir_data)
        self.context_history_len = context_history_len
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.save_hyperparameters('batch_size', 'accumulate_grad_batches', 
            'max_epochs', 'context_history_len', 'learning_rate',
            'warmup_proportion')
        self.save_hyperparameters(model.return_params())

        self.model = model
        
        self.loss = nn.BCEWithLogitsLoss()
        self.dict_acc = {
            k:pl.metrics.classification.Accuracy() for k in ["train",
                "val","test"] }
        self.dict_prec = {
            k:pl.metrics.classification.Precision() for k in ["train",
                "val","test"] }
        self.dict_recall =  {
            k:pl.metrics.classification.Recall() for k in ["train",
                "val","test"] }
                

        self.create_data_loaders()

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
        parser.add_argument('--config_file', default=None, help="Path to the \
            model hyperameters used in this model")        
        parser.add_argument('--dir_data', default="./combined_data", help="Relative directory path for datafiles")
        parser.add_argument('--gpus', default=None)
        parser.add_argument('--max_epochs', default=30)
        parser.add_argument('--accumulate_grad_batches', default=1)
        parser.add_argument('--context_history_len', default=1)
        parser.add_argument('--batch_size', default=2)
        parser.add_argument('--learning_rate', default=1e-4)
        parser.add_argument('--warmup_proportion', default=0.15)
        #parser.add_argument('--')
        #parser.add_argument('--default_root_dir', default=utils.get_path("./models/") )

        tparams = parser.parse_args()
        if tparams.config_file != None:
            tparams = json.load(open(utils.get_path(path)),"r" )

        return tparams

    def step(self, batch, step_name="train"):
        target = batch.pop('da')
        input_= batch

        output = self.forward(input_)
        loss = self.loss( output, target)
        
        loss_key = f"{step_name}_loss"
        
        self.log(loss_key, loss, True)
        self.dict_acc[step_name](output, target)
        self.dict_prec[step_name](output, target)
        self.dict_recall[step_name](output, target)
        
        if step_name == 'train':
            self.log(f'{step_name}_acc', self.dict_acc[step_name], on_step=True, on_epoch=False)
            self.log(f'{step_name}_rec', self.dict_recall[step_name], on_step=True, on_epoch=False)
            self.log(f'{step_name}_prec', self.dict_prec[step_name], on_step=True, on_epoch=False)
            
            return  { "loss": loss }
        
        else:
            self.log(f'{step_name}_acc', self.dict_acc[step_name], on_step=False, on_epoch=True)
            self.log(f'{step_name}_rec', self.dict_recall[step_name], on_step=False, on_epoch=True)
            self.log(f'{step_name}_prec', self.dict_prec[step_name], on_step=False, on_epoch=True)
            return {loss_key: loss}
        #return self.log(loss_key, loss, on_step=True, on_epoch=True, prog_bar=True, 'log':tensorboard_logs)

    def forward(self, input_, *args):
        return self.model(input_, *args)

    def training_step(self, batch, batch_idx):
        return self.step(batch,"train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def validation_epoch_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def test_epoch_end(self, outputs: List[dict]):
        loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss", loss)

    def create_data_loaders(self, shuffle=False):
        dir_train_set = os.path.join(self.dir_data,"train") #"./combined_data/train/"
        dir_val_set = os.path.join(self.dir_data,"val")
        dir_val_set = os.path.join(self.dir_data,"test")
        batch_size = 2

        dg = DataLoaderGenerator(dir_train_set, dir_val_set,
            dir_val_set, batch_size, self.model.tokenizer )
        
        self.train_dl, self.val_dl, self.test_dl = dg()

    def accuracy_calculator(self, outputs, targets):
        #Accuracy = Total Correct Observations / Total Observations

        return True
    
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    @lru_cache()
    def total_steps(self):
        train_files = [name for name in glob.glob(os.path.join(self.dir_data,"train","*")) ]

        utterance_count = sum( [ sum(1 for line in open(f,'r')) for f in train_files ] )

        conv_count = len(train_files)

        train_size = utterance_count - conv_count*self.context_history_len

        return ( train_size // self.batch_size*self.accumulate_grad_batches ) * self.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = self.total_steps()
        warmup_steps = int( self.total_steps() * self.warmup_proportion )

        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles = 3
            )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

class DaNet(nn.Module):
    """Transformer Based Model for DA classfication Task
    """

    def __init__(self, freeze_transformer=True, dropout=0.1, 
        base_model_name='microsoft/DialoGPT-small', model_name="DaNet", **kwargs):

        super(DaNet, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 12

        # Instantiate BERT model
        self.base_model_name = base_model_name       
        dict_transformertokenizer = utils.load_pretrained_transformer(self.base_model_name , transformer=True, tokenizer=True)
        self.transformer = dict_transformertokenizer['transformer']
        self.tokenizer = dict_transformertokenizer['tokenizer']
        self.freeze_transformer = freeze_transformer
        self.dropout = dropout
        # Freeze the Transformer model
        if self.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(D_in, H),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Linear(H, D_out)
        )
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
        
        parser.add_argument('--config_file', default=None, help="Path to the \
            training hyperameters for this model ")
        
        parser.add_argument('--base_model_name', default='bert-base-cased', required=False)
        parser.add_argument('--dropout', default=0.05, required=False, help="dropout")
        parser.add_argument('--freeze_transformer', default=True, required=False)
        parser.add_argument('--model_name', default='DaNet', required=False)
        
        mparams = parser.parse_known_args( )[0]
        if mparams.config_file != None:
            mparams = json.load(open(utils.get_path(path)),"r" )

        return mparams

    def forward(self, input_, *args):
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
        #input_ids, attention_mask, token_type_ids = input_
        input_ids = torch.squeeze(input_['input_ids'])
        attention_mask = torch.squeeze(input_['attention_mask'])
        token_type_ids = torch.squeeze(input_['token_type_ids'])

        # Feed input to BERT
        outputs = self.transformer(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        #cls_for_nsp = outputs[1][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

    def return_params(self):
        params = {}
        params['base_model_name'] = self.base_model_name
        params['freeze_transformer'] = self.freeze_transformer
        params['model_name'] = self.base_model_name
        params['dropout'] = self.dropout
        return params

class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """
    def __init__(self, dir_train_set, dir_val_set,
                    dir_test_set, batch_size,
                    tokenizer, 
                    context_history_len=1,
                    workers=0, **kwargs):

        self.dir_train_set = dir_train_set
        self.dir_val_set = dir_val_set
        self.dir_test_set = dir_test_set

        self.tokenizer = tokenizer
        label_mapping = json.load(open(utils.get_path("./label_mapping.json"),"r"))     
        self.target_binarizer = sklp.MultiLabelBinarizer()
        self.target_binarizer.fit( [label_mapping['MCONV']['labels_list'] ] )

            #Add a utility code that downloads pretrained bert tokenizer if it is not already in the relatively directory above
        self.bs = batch_size
        self.context_history_len = context_history_len
        self.workers  = workers

    def prepare_datasets(self):
        """prepares a train, validation and test set

        Returns:
            [type]: [description]
        """
        dir_sets = [self.dir_train_set, self.dir_val_set, self.dir_test_set]
        li_shuffle = [True, False, False]
        dataloaders = []
        
        dataloaders = [self.prepare_dataset(_dir, shuffle) for _dir,shuffle in zip(dir_sets,li_shuffle)]
        return dataloaders

    def prepare_dataset(self, dir_dset, shuffle=False):
        """Prepares a dataloader given a directory of text files each containing one conversation

        Args:
            dir_dset ([type]): [description]
        """
        files = glob.glob( os.path.join(dir_dset,"*") )
        random.shuffle(files)
        li_dsets = [ SingleDataset(_f, self.tokenizer, self.target_binarizer, 
            self.context_history_len) for _f in files ]

        concat_dset = torch.utils.data.ConcatDataset(li_dsets)
        dataloader = torch.utils.data.DataLoader(concat_dset, batch_size=self.bs,
            shuffle=shuffle, num_workers=self.workers, collate_fn=default_collate)
        
        return dataloader

    def __call__(self):
        train_dl, val_dl, test_dl = self.prepare_datasets()
        return train_dl, val_dl, test_dl
    
class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

    """
    def __init__(self, file_path, tokenizer, target_binarizer, context_history_len  ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.target_binarizer = target_binarizer
        self.context_history_len = context_history_len

    #def parse_file(self, file_path):
        with open(self.fp, 'r') as f:
            self.data = pd.read_csv(file_path, sep='|', header=0)
                    
    def __len__(self):
        return len(self.data) - self.context_history_len
    
    def __getitem__(self, index):
        datum = self.data[index:index+1+self.context_history_len]
        speaker, utterances, da = datum.T.values
        
        encoded_input = self.encode_tokenize( utterances.tolist() )
        binarized_target = self.target_binarizer.transform( [ da[-1].split() ] )

        map_datum = {**encoded_input, 'da':torch.squeeze(torch.from_numpy(binarized_target.astype(np.float))) }
        return map_datum

    def encode_tokenize(self, li_str):
        #_str_tknized = self.tokenizer.tokenize(_str)
        
        encoded_input = self.tokenizer(*li_str, add_special_tokens=True, padding='max_length', 
            truncation=True, max_length=160, return_tensors='pt', return_token_type_ids=True )
                
        return encoded_input


def main(tparams, mparams):
    gc.collect()
    torch.cuda.empty_cache()

    version_name = utils.get_version_name(mparams.model_name)
    utils.save_version_params(tparams, mparams, version_name )
    
    # Setting up callbacks and loggers
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='auto'
    )
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=3, mode='min', 
        dirpath=utils.get_path(f'./models/{version_name}/logs') )

    tb_logger = pl_loggers.TensorBoardLogger(utils.get_path(f'./models/{version_name}/logs'))
    
    # Setting up model, training_module and Trainer
    danet = DaNet(**vars(mparams))

    training_module = TrainingModule(**vars(tparams), model=danet )
    #training_module.save_hyperparameters(vars(mparams))

    trainer = pl.Trainer.from_argparse_args(tparams, progress_bar_refresh_rate=5,
                     callbacks=[early_stop_callback,checkpoint_callback],
                     check_val_every_n_epoch=1, logger=tb_logger,
                     default_root_dir=utils.get_path(f"./models/{version_name}"),
                     fast_dev_run=True, 
                     #log_gpu_memory=True
                     )

    trainer.fit(training_module)
    trainer.test()

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 
    #parent_parser2 = argparse.ArgumentParser(add_help=False)    
    
    parser_program = parent_parser.add_argument_group("program")

    # add PROGRAM level args
    #parser_program.add_argument('--conda_env', type=str, default='some_name')
    #parser_program.add_argument('--notification_email', type=str, default='johnsmith@email.com')
    
    # add model specific args
    mparams = DaNet.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = TrainingModule.parse_train_specific_args(parent_parser)

    main(tparams, mparams)