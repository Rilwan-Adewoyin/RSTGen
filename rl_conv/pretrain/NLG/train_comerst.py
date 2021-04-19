#Script for training models related to predicting Key phrases from knowledge graph
# Example of how to use batch generate -> https://github.com/allenai/comet-atomic-2020/blob/master/models/comet_atomic2020_bart/generation_example.py

# Use this for finishing off code for fine-tuning https://github.com/allenai/comet-atomic-2020/blob/master/models/comet_atomic2020_bart/finetune.py

import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import utils_comerst as utils

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

#region COMET model
class Comet:

    def __init__(self, model_path):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarisation" #use this to define what version of the model you are using
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(self.chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs
    
    def chunks(self, lst, n):
    """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

class TrainingModule(pl.LightningModule):

    def __init__(self, model_params, batch_size=20, 
                    dir_data=None, 
                    accumulate_grad_batches=1,
                    max_epochs=25,
                    gpus=1, 
                    learning_rate=1e-3,
                    warmup_proportion=0.1,
                    workers=0,
                    lr_schedule='hard_restarts',
                    mode = 'train_new',
                    data_splits = {'train':0.6,'val':0.2,'test':0.2},
                    inference_context_utt = None, #amount of words from utterance to use as context
                    optimizer_type="AdamW",
                    tag='',
                    *args,
                    **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        self.model = NLG( **model_params )
        self.mode = mode
        self.workers = workers
        self.data_splits = data_splits
        self.optimizer_type = optimizer_type
        
        
        if self.mode in ['train_new','train_cont','test']:
            self.dir_data = utils.get_path(dir_data)
            self.inference_context_utt = inference_context_utt
            self.create_data_loaders( )
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

        if self.mode in ['train_new','train_cont']:
            self.max_epochs = max_epochs
            self.warmup_proportion = warmup_proportion
            self.lr_schedule = lr_schedule
            self.learning_rate = learning_rate
        
            train_params_to_save = self.return_params()
            model_params_to_save = self.model.return_params()

            self.hparams = { **train_params_to_save, **model_params_to_save}

            self.inference_samples = list( islice( self.inference_dl, 10 ) )
            bad_words = ["<|rst|>","<|ta|>",r"\n" ] 
            bad_words_ids = [self.model.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=False) for bad_word in bad_words]
            bad_words_ids = [self.model.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words]
            bad_words_ids = bad_words_ids + [[526], [55],[8172]]
            

            generation_params = {'num_beams':1, 'temperature':1.1, 'repitition_penalty':1.2, 
                                'top_k': 50, 'top_p':0.85,
                                'length_penalty':1.5, 'early_stopping':True,
                                'do_sample':True, 'bad_words_ids':bad_words_ids, 'no_repeat_ngram_size':3
                                ,'min_length':5, 'max_length':80  } 
                                
                                # 'max_length':self.model.nlg_tokenizer.max_input_len  } 
            
            self.inference_generation_params = generation_params

            del self.inference_dl

        if self.mode in ['inference']:
            self.eval() 
            self.freeze() 

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset/reddit_large_annotated_long3", help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('-me','--max_epochs', default=32, type=int)
        parser.add_argument('-agb','--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('-bs','--batch_size', default=5, type=int)
        parser.add_argument('-lr','--learning_rate', default=5e-4, type=float)
        parser.add_argument('--warmup_proportion', default=0.15)
        parser.add_argument('--workers', default=16, type=int) #TODO: change to 6
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference'])
        parser.add_argument('--lr_schedule', default='cosine_warmup', required=False, choices =['cosine_warmup','LROnPlateau','hard_restarts','constant'])
        parser.add_argument('--splits', default={'train':0.6,'val':0.2,'test':0.2}, required=False, type=str )
        parser.add_argument('--version', default=None,required=False, type=int, help="The Experimental Versioning for this run" )
        parser.add_argument('--precision', default=16,required=False, type=int, help="Precision to use", choices=[16,32] )
        parser.add_argument('-opt','--optimizer_type', default="AdamW",required=False, type=str, help="Optimizer to use", choices=["AdamW","Adafactor"] )
        parser.add_argument('--tag',default='',required=True, type=str)
        parser.add_argument('--override',default=False, type = lambda x: bool(int(x)), choices=["0","1"] )
        parser.add_argument('--inference_context_utt', default=4, type=int)
            #TODO: check --version of required type None actually works
        tparams = parser.parse_known_args()[0]
        #tparams.splits = json.loads(tparams.splits)

        return tparams
    
    @staticmethod
    def instatiate_training_module( tparams=None, mparams=None ):
        """Create training module

        Args:
            tparams ([type]): [description]
        """
        #pl.seed_everything(10)

        if tparams['mode'] in ["train_new"]:
            training_module = TrainingModule(**tparams, model_params=mparams  )
            
        elif tparams['mode'] in ["train_cont", "inference"]:
            
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #restore/update param files from the checkpoint
            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                tparams.update ( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'batch_size', 'lr_schedule', 'learning_rate','precision','splits','optimizer_type','tag']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name','loss_type','model_name','fda','frst','ftopic','max_input_len',
                    'frst_version','scale_grad_by_freq','freeze_pretrained']} )
                
                mparams_json = {k:json.loads(v) for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'context_len'] }
        
                mparams =  {**mparams, **mparams_json}
            
            else:
                print("param files not found utilsing default or user entered params\n")
                
            #Restore/update Training Module
            training_module = TrainingModule(**tparams, model_params=mparams)
            training_module.load_state_dict(checkpoint['state_dict'])

        elif tparams['mode'] in ["test"]:
            
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #restore/update param files from the checkpoint
            try:
                tparams.update ( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'lr_schedule', 'learning_rate','precision','splits','optimizer_type']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name','loss_type','model_name','fda','frst','ftopic','max_input_len']} )
            except KeyError:
                pass
            
            #Restore/update Training Module
            training_module = TrainingModule(**tparams, model_params=mparams)
            training_module.load_state_dict(checkpoint['state_dict'])

        else:
            raise ValueError("tparams['mode'] must be in range [train_new, train_cont, test, inference]")

        return training_module

    @staticmethod
    def instatiate_trainer( tparams, tb_logger, training_module):
        """[summary]

            Creates The Trainer and callbacks
        """
        dir_checkpoints = tparams['dir_checkpoints']
        
        # Creating Callbacks
        callbacks = []        
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=2, 
            mode='min', dirpath=dir_checkpoints, 
            filename='{epoch:03d}_{val_loss:.5f}')
        
        checkpoint_callback._save_model  = types.MethodType(monkey_save_model,checkpoint_callback) #monkey patch
        checkpoint_callback._monitor_candidates = types.MethodType(_monitor_candidates, checkpoint_callback) # monkey patch

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)

       
        if tparams['gpus'] in [0,1]:
            accelerator=None
        else:
            accelerator = 'ddp'

        
        if tparams['mode'] in ["train_new"]:
            
            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                        progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                        default_root_dir=tparams['dir_checkpoints'],
                        check_val_every_n_epoch=1, logger=tb_logger,
                        #log_every_n_steps=20,
                        precision=tparams['precision'], callbacks=callbacks,
                        #accelerator='ddp2', amp_level='O2',# use_amp=True,
                        accelerator=accelerator,
                        #limit_train_batches =10,
                        #limit_val_batches = 10,
                        val_check_interval=0.2,
                        num_sanity_val_steps=0, 
                        #track_grad_norm = True,
                        #overfit_batches=25,
                        #fast_dev_run=2, 
                        #log_gpu_memory=True
                        )

        elif tparams['mode'] in ["train_cont","inference"]:
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #training_module.load_state_dict(checkpoint['state_dict'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                    check_val_every_n_epoch=1, logger=tb_logger,
                    log_every_n_steps=20,   
                    precision=tparams['precision'],
                    callbacks=callbacks,
                    #accelerator='ddp2',  amp_level='O2', # use_amp=True,
                    accelerator=accelerator,
                    #limit_train_batches = 0.4,
                    #val_check_interval=0.5,
                    #limit_val_batches = ,
                    val_check_interval=0.2,
                    num_sanity_val_steps=0,
                    #track_grad_norm = True,
                    #overfit_batches=5
                    #,fast_dev_run=2, 
                    #log_gpu_memory=True
                    )

            # load callback states
            trainer.on_load_checkpoint(checkpoint)
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

        elif tparams['mode'] in ["test"]:
            
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            training_module.load_state_dict(checkpoint['state_dict'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                    check_val_every_n_epoch=1,
                    checkpoint_callback=False,
                    logger=tb_logger,
                    log_every_n_steps=1,   
                    precision=tparams['precision'],
                    )

            # load callback states
            trainer.on_load_checkpoint(checkpoint)
           

        return trainer,training_module
    
    @staticmethod
    def get_ckpt_file(_dir_checkpoint,mode='best'):
        if mode=='best':
            checkpoint_yaml_file = os.path.join( _dir_checkpoint,"best_k_models.yaml" )
            scores_dict = yaml.load( open(checkpoint_yaml_file,"r") ) #key= ckptpath, value = val_loss
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if torch.cuda.is_available():
                #checkpoint = torch.load(best_ckpt_path, map_location=lambda storage, loc: storage) )
                #checkpoint = torch.load(best_ckpt_path, map_location=lambda storage, loc: storage.cuda())  
                checkpoint = torch.load(best_ckpt_path, map_location='cpu' )  

            else:
                checkpoint = torch.load(best_ckpt_path, map_location='cpu')            
        else:
            raise NotImplementedError
        
        return checkpoint
        
    @staticmethod
    def start(trainer, tparams,training_module, mparams ):
        
        if tparams['mode'] in ['train_new','train_cont']:    
            trainer.fit(training_module )
        
        if tparams['mode'] in ["test"]:
            
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])
            training_module.load_state_dict(checkpoint['state_dict'])

            #trainer.on_load_checkpoint(checkpoint)
            training_module.eval() 
            training_module.freeze() 
            
            dict_results = trainer.test(test_dataloaders=training_module.test_dl, model = training_module)

            #Saving test results for model to file
            _dir = os.path.join(tparams['model_dir'], mparams['model_name'])
            fn = os.path.join(_dir,"results.json")

            if os.path.isfile(fn) == False:
                existing_results = {}
            else:
                with open( fn, 'r' ) as outfile:
                    existing_results = json.load( outfile )

            existing_results[ f"{mparams['model_name']}_{tparams['version']}" ] = dict_results[0]['test_loss']
            
            with open( fn, 'w' ) as outfile:
                json.dump( existing_results, outfile)
                       
        elif tparams['mode'] in ['infernece']: 
            training_module.eval() 
            training_module.freeze() 
            raise NotImplementedError   
    
    @staticmethod
    def load_nlgmodel(model_name="NLG_rt", model_version=11,max_input_len=None):
        # Loading in NLG model
        checkpoint = TrainingModule.get_ckpt_file(f'./models/{model_name}/version_{model_version}/checkpoints')

        # Getting tparams
        tparams = {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
            'batch_size', 'lr_schedule', 'learning_rate','precision','splits','optimizer_type',
            'tag']}

        tparams['mode'] = 'inference'

        mparams =  {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
            'base_model_name','loss_type','model_name','fda','frst','ftopic','max_input_len',
            'freeze_pretrained','frst_version','scale_grad_by_freq']}
        
        if model_version in [14,15,16]:
            mparams_json = {'context_len': {'rst':16, 'topics':30} }
        else:
            mparams_json = {k:json.loads(v) for k,v in checkpoint['hyper_parameters'].items() if k in [
            'context_len'] }

        mparams =  {**mparams, **mparams_json}
        
        if max_input_len != None:
            mparams['max_input_len'] = max_input_len
            
        # Loading Training Module
        training_module = TrainingModule(**tparams, model_params=mparams )
        training_module.load_state_dict(checkpoint['state_dict'])
        nlg_model = training_module.model

        # Deleting checkpoints to free up GPU space
        del checkpoint
        torch.cuda.empty_cache()
          
        if torch.cuda.is_available():
            nlg_model =nlg_model.cuda()
        
        return nlg_model

    @staticmethod
    def model_name(mparams):

        #Adapting Model Name to handle testing of different scenarios
        if mparams['fda'] and mparams['frst'] and mparams['ftopic']:
            mparams['model_name'] = f"{mparams['model_name']}_drt"
        
        elif not mparams['fda'] and mparams['frst'] and mparams['ftopic']:
            mparams['model_name'] = f"{mparams['model_name']}_rt"
        
        else:
            NotImplementedError

    @auto_move_data
    def forward(self, input_):
        return self.model(input_)

    def step(self, batch, step_name):
        
        input_= batch
        _, loss = self.forward(input_) #(lm_logits and loss)
        loss_key = f"{step_name}_loss"
        
        output = {}

        if step_name == 'train':
            output["loss"] = loss

        else:
            str_loss_key = loss_key       
            self.log( str_loss_key, loss)
            
            output[str_loss_key]=loss


        return  output 
        
    def training_step(self, batch, batch_idx):
        output = self.step(batch,"train")
        return output

    def validation_step(self, batch, batch_idx):
        output = self.step(batch, "val")
        return output

    def test_step(self, batch, batch_idx):
        output = self.step(batch, "test")
        return output

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
            loss = torch.stack([x[f"{step_name}_loss"] for x in outputs]).mean()
            self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True)
                           
    def create_data_loaders(self, shuffle=False, **kwargs):
       
        dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.model.nlg_tokenizer, 
                workers=self.workers, mode=self.mode, split=self.data_splits,
                fda=self.model.fda, frst=self.model.frst,
                ftopic=self.model.ftopic,
                inference_context_utt=self.inference_context_utt)

        _dict_dl = dg()
        self.train_dl = _dict_dl['train_dl']
        self.val_dl = _dict_dl['val_dl']
        self.test_dl = _dict_dl['test_dl']
        self.inference_dl = _dict_dl['inference_dl']

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    @lru_cache()
    def total_steps(self):

        ds_size = len(self.train_dl) // self.gpus
        steps = (ds_size * self.max_epochs) // (self.accumulate_grad_batches)
        return steps

    def configure_optimizers(self):
        
        if self.optimizer_type == "AdamW":
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            
            warmup_steps = int( self.warmup_proportion*self.total_steps() )

            lr_schedule = get_cosine_schedule_with_warmup(optimizer, 
                            warmup_steps, self.total_steps(), 0.5 )

            return [optimizer], [{ "scheduler":lr_schedule ,"interval": "step", "monitor":"val_loss"}]
        
        elif self.optimizer_type == "Adafactor":
            optimizer = torch.optim.Adafactor(
                self.model.parameters(), lr=self.learning_rate,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=True,
                warmup_init=False
                )

            return [optimizer]
            raise NotImplementedError

    def return_params(self):
        params = {}
        keys = ['batch_size','accumulate_grad_batches','lr_schedule','learning_rate','max_epochs','dir_data'
            'warmup_proportion','optimizer_type','tag','inference_context_type']
        
        params = {
            k:self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }


        return params


class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """
    def __init__(self, dir_data, batch_size,
                    tokenizer, 
                    workers=0, mode='train_new',
                    splits={'train':0.6,'val':0.2,'test':0.2},
                    fda=True, frst=True, ftopic=True,
                    inference_context_utt=0,
                    **kwargs):
        
        self.dir_data = dir_data
        self.tokenizer = tokenizer
        self.splits = splits

        self.bs = batch_size
        self.workers  = workers
        self.mode = mode

        self.fda = fda
        self.frst = frst 
        self.ftopic = ftopic
        
        self.inference_context_utt = inference_context_utt

    def prepare_dataloaders(self):
        """prepares a train, validation and test set

        Returns:
            [type]: [description]
        """
                
        if self.mode in [ 'train_new', 'train_cont']:
            train_dl = self.prepare_dataloader(self.dir_data, shuffle=True, split_name='train' )
            val_dl = self.prepare_dataloader(self.dir_data, shuffle=False,split_name='val'  )
            test_dl = self.prepare_dataloader(self.dir_data, shuffle=False,split_name='test'  )
            inference_dl = self.prepare_dataloader(self.dir_data, shuffle=True, split_name="inference")
        
        elif self.mode in ['test']:
            train_dl= None
            val_dl = None
            test_dl = self.prepare_dataloader(self.dir_data, shuffle=False ,split_name='test' )
            inference_dl = None

                    
        dict_dl = {'train_dl':train_dl,
                    'val_dl':val_dl,
                    'test_dl':test_dl,
                    'inference_dl':inference_dl}

        return dict_dl 

    def prepare_dataloader(self, dir_data, shuffle=False, 
        split_name='train'):

        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """
        #getting all files from all different subreddits/types of conversation
        fns = glob.glob(  os.path.join( utils.get_path(dir_data),"*","*") )
        fns = [fn for fn in fns if os.path.split(fn)[-1]!="lock"]
        #getting number of utterances records in each file
        files_sizes = [ int(fn[-10:]) for fn in fns]

        #defining starting line and total lines to use for dataset
        if split_name == 'train':
            line_starts = [0]*len(files_sizes)
            line_ends = [ ls+int(fs*self.splits['train']) for ls,fs in zip(line_starts, files_sizes)  ]
            shuffle = True
            ifc = 0
        
        elif split_name == 'val':
            line_starts = [ int(fs*self.splits['train']) for fs in files_sizes  ]
            line_ends = [ ls+int(fs*self.splits['val']) for ls,fs in zip(line_starts, files_sizes)  ]
            shuffle = False
            ifc = 0

        elif split_name == 'test':
            line_starts = [ int(fs*(1-self.splits['test']) ) for fs in files_sizes  ]
            line_ends = files_sizes
            shuffle = False
            ifc = 0

        elif split_name == 'inference':
            line_starts = [ random.randrange( int(fs*(1-self.splits['test'])), fs) for fs in files_sizes  ]
            line_ends =  files_sizes
            shuffle = False
            ifc = self.inference_context_utt

        li_dsets = [ SingleDataset(_f, self.tokenizer, line_start, line_end, self.fda, self.frst, self.ftopic, ifc) 
                        for _f, line_start, line_end in zip(fns, line_starts, line_ends) ]

        if split_name == 'inference':
            random.sample(li_dsets,10)
            bs = 1
        else:
            bs = self.bs

        concat_dset = torch.utils.data.ConcatDataset(li_dsets)

        dataloader = torch.utils.data.DataLoader(concat_dset, batch_size=bs,
            shuffle=shuffle, num_workers=self.workers, collate_fn=default_collate)
        
        return dataloader

    def __call__(self):
        dict_dl = self.prepare_dataloaders()
        return dict_dl
    


def main(tparams={}, mparams={}):

    mparams['model_name'] = Comet.model_name(mparams)
    
    # Defining Logger
    tb_logger = pl_loggers.TensorBoardLogger( 
                    save_dir = os.path.abspath(tparams['model_dir']),
                    name = mparams['model_name'],
                    version = tparams['version'] )
    tparams['version'] =  tb_logger.version
    
    tparams['dir_checkpoints'] = os.path.join(tparams['model_dir'],mparams['model_name'],f"version_{tparams['version']:02d}",'checkpoints' )
    
    os.makedirs(tparams['dir_checkpoints'],exist_ok=True)

    # initiating training loop
    training_module = TrainingModule.instatiate_training_module( tparams, mparams)
    trainer, training_module = TrainingModule.instatiate_trainer( tparams,  tb_logger, training_module)
    TrainingModule.start(trainer, tparams, training_module, mparams)

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False) 
    
    # add model specific args
    mparams = COMET.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = TrainingModule.parse_train_specific_args(parent_parser)

    if tparams.mode == "test":
        assert tparams.gpus in [0,1]

    # adjust to allow ddp to work on computer
    if tparams.gpus not in [0,1]:
        os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'
        os.environ['MASTER_PORT'] = '65302'

    main(vars(tparams), vars(mparams))