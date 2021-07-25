# This script is for fine-tuning the Combined Model on a specific task

from random import triangular
from nltk.tree import Tree
from torch.nn.modules.loss import CosineEmbeddingLoss
import train_comerst
import train_nlg
import train_rstplanner
import argparse
import ujson
from torch import nn
#os.environ['NCCL_SOCKET_IFNAME'] =  'lo' 
import json
import torch
import argparse
import os
from typing import Optional, Callable, Union, Optional, List, Iterable

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

class CombinedModel(nn.Module):

    # Generation method
        # Input Factors
            # 1)RST Tree:kwargs used by rst_planner
            # 2)Key_phrase: key phrases and (possibly) their position and (possibly) their Nuclearity
            # 3)NLG:
        # Prediction Order
            # All of RST Tree
            # All key phrases
            # Sequentially predict text, using previous text as word context and rst tree and key phrases to guide

    # Set up sub-model loading
    # Set up generation methods / Test in Jupyter
    # Set up dataloader and training script

    def __init__(self, comerst_mv=13, nlg_mv=15,device="cuda:0", rst_init_params={},  ):
        
        
        self.comerst = train_comerst.load_comerst( model_version=comerst_mv , device=device )
        self.nlg = train_nlg.load_nlgmodel(model_version=nlg_mv, device = device )
        self.rstplanner = train_rstplanner.RSTPlanner(**rst_init_params)
        
    
    def forward(self,  ):
        """Produces a chunk of text"""
        return 
    
    def generate(self, rst_gen_params={}, kp_gen_params={} ):
        """Generates a whole text"""
        
        # sample an rst template
        rst_sampling_params = sampling_params
        rst_context = self.sample_rst( rst_gen_params ) 

        # sample key phrases 
        kp_gen_params = 
        key_phrases = self.sample_keyphrases( kp_gen_params, rst_context)

        # generate text
        generated_text = self.gen_text()
    
    
    def sample_rst(self, sampling_params):

        rst_chain = self.train_rstplanner.sample_rst_chain( **sampling_params )

        rst_chain_decoded = self.train_rstplanner.deserialize_chain( )
    
    def sample_key_phrase(self, ):
        pass

    def gen_text(self, ):
        pass


    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        
        parser.add_argument('--mv_combined', default=11, required=True)
        parser.add_argument('--mv_nlg', default=None, required=False)
        parser.add_argument('--mv_comerst', default=None, required=False)
        parser.add_argument('')
        

        mparams = parser.parse_known_args( )[0]
       
        return mparams

class TrainingModule(pl.LightningModule):

    def __init__(self, mparams, batch_size=20, 
                    dir_data=None,
                    accumulate_grad_batches=1,
                    max_epochs=10,
                    gpus=1, 
                    learning_rate=1e-4,
                    warmup_proportion=0.1,
                    workers=0,
                    mode = 'train_new',
                    data_splits = {'train':0.6,'val':0.2,'test':0.2},
                    tag='',
                    *args,
                    **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        self.model = CombinedModel( **mparams )
        
        self.mode = mode
        self.workers = workers
        self.data_splits = data_splits
        
        
        if self.mode in ['train_new','train_cont','test']:
            self.dir_data = dir_data
            self.create_data_loaders(  )
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

        if self.mode in ['train_new','train_cont']:
            self.max_epochs = max_epochs 
            self.warmup_proportion = warmup_proportion
            self.learning_rate = learning_rate
        
            train_params_to_save = self.return_params()
            model_params_to_save = self.model.return_params()

            self.hparams.update({ **train_params_to_save, **model_params_to_save})
            self.save_hyperparameters({ **train_params_to_save, **model_params_to_save})

            self.inference_samples = list( islice( self.inference_dl, 10 ) )
            del self.inference_dl

        if self.mode in ['inference']:
            self.eval() 
            self.freeze() 

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset_keyphrase_v2", help="Relative directory path of rst data")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=28, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('-s','--batch_size', default=100, type=int)
        parser.add_argument('-l','--learning_rate', default=1e-5, type=float)
        parser.add_argument('--warmup_proportion', default=0.1)
        parser.add_argument('--workers', default=12, type=int) 
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference'])
        parser.add_argument('--splits', default={'train':0.6,'val':0.2,'test':0.2}, required=False, type=str )
        parser.add_argument('--version', default=0,required=False, type=int, help="The Experimental Versioning for this run" )
        parser.add_argument('--precision', default=16,required=False, type=int, help="Precision to use", choices=[16,32] )
        parser.add_argument('--tag', default='default model', required=False, type=str)
        
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

        if   tparams['mode'] in ["train_new"]:
            training_module = TrainingModule(**tparams, model_params=mparams  )
            
        elif tparams['mode'] in ["train_cont", "inference"]:
            
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #restore/update param files from the checkpoint
            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                
                tparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in 
                    ['learning_rate','precision','splits','tag','loss_weight_rst','loss_weight_comet'] } )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_tokenizer_name', 'model_name', 'max_len_head', 'max_len_tail',
                    'scale_grad_by_freq','filter_atomic_rels','max_edu_nodes_to_select',
                    'relation_embedding' ]} )
                
                mparams_json = {k:json.loads(v) for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'dict_embed_mnorms'] }
        
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
                    'learning_rate','precision','splits','loss_weight_rst','loss_weight_comet']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_tokenizer_name','loss_type','model_name','max_len_head','max_len_tail']} )
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
        # checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=2, 
        #     mode='min', dirpath=dir_checkpoints, 
        #     filename='{epoch:03d}_{val_loss:.5f}')
        #NOTE: debugging
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=0, 
            mode='min', dirpath=dir_checkpoints, 
            filename='{epoch:03d}_{val_loss:.5f}')
        
        
        checkpoint_callback._save_model  = types.MethodType(utils.monkey_save_model, checkpoint_callback) #monkey patch

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            #monitor='val_loss_comet',
            min_delta=0.00,
            patience=5,
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
                        #check_val_every_n_epoch=1,
                        logger=tb_logger,
                        #log_every_n_steps=20,
                        precision=tparams['precision'], callbacks=callbacks,
                        #accelerator='ddp2', amp_level='O2',
                        accelerator=accelerator,
                        limit_train_batches =10,
                        limit_val_batches = 5,
                        #val_check_interval=0.3,
                        #num_sanity_val_steps=0, 
                        #overfit_batches=25,
                        reload_dataloaders_every_epoch=True,
                        multiple_trainloader_mode='max_size_cycle'
                        #,gradient_clip_val=0.00001, gradient_clip_algorithm='value'
                        )

        elif tparams['mode'] in ["train_cont","inference"]:
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                    #check_val_every_n_epoch=1,
                    logger=tb_logger,
                    #log_every_n_steps=20,   
                    precision=tparams['precision'],
                    callbacks=callbacks,
                    #accelerator='ddp2',  amp_level='O2', # use_amp=True,
                    accelerator=accelerator,
                        limit_train_batches =20,
                        limit_val_batches = 5,
                    #val_check_interval=0.3,
                    num_sanity_val_steps=0,
                    #track_grad_norm = True,
                    #overfit_batches=5
                    #,fast_dev_run=2, 
                    #log_gpu_memory=True,
                    reload_dataloaders_every_epoch=True,
                    multiple_trainloader_mode='max_size_cycle'
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
            ckpt_exists =  os.path.exists(checkpoint_yaml_file)
            if ckpt_exists == False:
                return False
            
            scores_dict = yaml.load( open(checkpoint_yaml_file,"r"), Loader = yaml.FullLoader ) #key= ckptpath, value = val_loss
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if os.path.exists(best_ckpt_path) == False:
                root_dir = Path(__file__).resolve().parents[4]
                best_ckpt_path = os.path.join( str(root_dir), best_ckpt_path[ best_ckpt_path.index('mastering-conversation'): ] )

            if torch.cuda.is_available():
                checkpoint = torch.load(best_ckpt_path, map_location='cpu' )  

            else:
                checkpoint = torch.load(best_ckpt_path, map_location='cpu')            
        else:
            raise NotImplementedError
        
        return checkpoint
        
    @staticmethod
    def start(trainer, tparams, training_module, mparams ):
        
        if tparams['mode'] in ['train_new','train_cont']:    
            trainer.fit( training_module )
        
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
    def load_combinedmodel(model_version ,model_name="CombinedModel",  device="cuda:0", mv_nlg=None, mv_comerst=None ):
        # Loading in CombinedModel
        checkpoint = TrainingModule.get_ckpt_file(f'./models/{model_name}/version_{model_version}/checkpoints')

        if checkpoint == None:
            #make new model
            assert mv_nlg!=None and mv_comerst!=None
            tparams = TrainingModule.parse_train_specific_args()
            mparams = {'mv_nlg':mv_nlg, 'mv_comerst':mv_comerst }
            training_module = TrainingModule(**tparams, model_params=mparams )
        
        else:
            # Getting tparams
            tparams = {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                'precision','splits','tag']}
            tparams['mode'] = 'inference'

            mparams =  {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                'model_name','mv_nlg','mv_comerst']}
        
            mparams_json = {k:json.loads(v) for k,v in checkpoint['hyper_parameters'].items() if k in [] }

            mparams =  {**mparams, **mparams_json}
                    
            # Loading Training Module
            training_module = TrainingModule(**tparams, model_params=mparams )
            training_module.load_state_dict(checkpoint['state_dict'])
            del checkpoint
            torch.cuda.empty_cache()        
        model = training_module.model

        if device != 'cpu' and torch.cuda.is_available():
            model = model.to(device)
        
        return model

    @staticmethod
    def model_name(mparams):
        return mparams['model_name']

    @auto_move_data
    def forward(self, input_):
        return self.model(input_)

    def step(self, batch, step_name):
        
        input_ = batch

        model_output = self.forward(input_) #(lm_logits and loss)
        loss = None
        
        if 'comet' in model_output:
            lm_loss_comet = model_output['comet'].loss * self.loss_weight_comet
            loss = lm_loss_comet
        
        if 'rst' in model_output:
            lm_loss_rst = model_output['rst'].loss *self.loss_weight_rst
            if loss == None:
                loss = lm_loss_rst
            else:
                loss = loss + lm_loss_rst
        
        loss_key = f"{step_name}_loss"
        
        output = {}
        if step_name == 'train':
            output["loss"] = loss
            
        else:       
            self.log( loss_key, loss)
            output[ loss_key ]=loss


        if 'rst' in model_output:
            self.log( loss_key+"_rst", lm_loss_rst )
        elif 'comet' in model_output:
            self.log( loss_key+"_comet", lm_loss_comet )
            # here output dictionary instead
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
            if len(outputs) > 0:
                loss = torch.stack([x[f"{step_name}_loss"] for x in outputs]).mean()
                self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True)

        if step_name == "val" and _get_rank() == 0 :
            bad_words = ['"']
            bad_words_ids = [ self.model.tokenizer.base_tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words ]
            
            generation_kwargs_comet = {'num_beams':1, 'temperature':1.2, 'repitition_penalty':1.0, 
                                'early_stopping':False, 'do_sample':False, 'no_repeat_ngram_size':3, 
                                'num_return_sequences':1, 'bad_words_ids':bad_words_ids,
                                'min_length':3, 'max_length':18 }

            generation_kwargs_rst = {'num_beams':1, 'temperature':1.2, 'repitition_penalty':1.0, 
                                'early_stopping':False, 'do_sample':False, 'no_repeat_ngram_size':3, 
                                'num_return_sequences':1, 'bad_words_ids':bad_words_ids,
                                'min_length':3, 'max_length':10 }
                    
                    # At end of validation loop produce some quantitative examples of model's performance

            # Making directory for inference testing results
            dir_infer = os.path.join(self.trainer.log_dir,"inference")
            if not os.path.exists(dir_infer):
                os.makedirs(dir_infer,exist_ok=True)
        
            # data holders
            li_comet_heads = []
            li_comet_rels = []
            li_comet_preds = []
            li_comet_tails = []

            li_rst_heads = []
            li_rst_rels = []
            li_rst_preds = []
            li_rst_tails = []

            for idx, batch in enumerate( self.inference_samples ):
                
                # COMET testing
                if 'comet' in batch:
                    batch_comet =       batch['comet']
                    batch_comet_heads = [ self.model.tokenizer.base_tokenizer.decode( head_ids, skip_special_tokens=True  ).strip() for head_ids in batch_comet['head_ids'] ]
                    batch_comet_rels =  [ self.model.tokenizer.atomic_rel_labeler.inverse_transform( rels_ids ) for rels_ids in batch_comet['rels_ids'].cpu().numpy().tolist() ]
                    batch_tails_comet = [  self.model.tokenizer.base_tokenizer.decode( tail_ids , skip_special_tokens=True  ).strip()  for tail_ids in batch_comet['tail_ids'] ]

                    preds = self.model.generate_from_batch( batch_comet, comet_or_rst="comet", generation_kwargs=generation_kwargs_comet )
                    
                    li_comet_heads.extend(batch_comet_heads)
                    li_comet_rels.extend(batch_comet_rels)
                    li_comet_tails.extend(batch_tails_comet)
                    li_comet_preds.extend(preds)

                # RST Testing
                if 'rst' in batch:
                    batch_rst = batch['rst']
                            
                    # RST -  prediction for every elem in batch
                    preds = self.model.generate_from_batch( batch_rst, comet_or_rst="rst", generation_kwargs=generation_kwargs_rst )
                                    
                    #batch decoding to get original data for each elem in batch
                    batch_rst_heads = [ self.model.tokenizer.base_tokenizer.decode( head_ids,  skip_special_tokens=True ).split('</s><s>') for  head_ids in batch_rst['head_ids']  ]
                    batch_rst_heads = [ [ _.strip("<s>").strip("</").strip() for _ in heads_rst ] for heads_rst in batch_rst_heads ]
                    
                    batch_heads_treepos_rst = batch_rst['head_treepos_ids'].cpu().numpy().tolist()
                    batch_heads_treepos_rst = [ [ key for key, group in groupby(treepos) ] for treepos in batch_heads_treepos_rst ]
                    
                    batch_edu_pos_for_head = batch_rst.get('li_edu_pos_for_head',None).cpu().numpy().tolist()
                    
                    batch_rels_ids_rst = [ self.model.tokenizer.rst_rel_labeler.inverse_transform_patch( rst_rel_ids ).tolist() for rst_rel_ids in batch_rst['rst_rel_ids'].tolist() ]                    
                    batch_rels_treepos_rst = [ id_ for id_ in batch_rst['rst_treepos_ids'].tolist() if id_!= self.model.embedding_rst_pos.padding_idx]
                    
                    batch_ns_rst = [ [ns for ns in rst_ns_ids if ns<len(self.model.tokenizer.rst_ns_labeler.classes_)  ] for rst_ns_ids in batch_rst['rst_ns_ids'].tolist() ]
                    batch_rels_ns_rst = [ self.model.tokenizer.rst_ns_labeler.inverse_transform( rst_ns_id ).tolist() for rst_ns_id in batch_ns_rst ]

                    batch_tails_rst = [self.model.tokenizer.base_tokenizer.decode( tail_id, skip_special_tokens=True ).strip() for tail_id in batch_rst['tail_ids'].tolist()  ]
                    
                    batch_edu_pos_for_tail = batch_rst.get('li_edu_pos_for_tail',None).cpu().numpy().tolist()
                            
                    # looping through every element in the batch to get head, rel, tail for recording
                    for idx1 in range(len(preds)):
                    
                        head = { ','.join(map(str,batch_edu_pos_for_head[idx1])):  batch_rst_heads[idx1][0] }       
                        rel = [ {pos:[rel, ns]} for pos, rel,ns in zip(batch_rels_treepos_rst[idx1], batch_rels_ids_rst[idx1], batch_rels_ns_rst[idx1]) ]
                        tail = {','.join(map(str,batch_edu_pos_for_tail[idx1])):batch_tails_rst[idx1].strip() }
                                                            
                        li_rst_heads.append( head )
                        li_rst_rels.append( rel )
                        li_rst_tails.append( tail )
                        li_rst_preds.append( preds[idx1] )
                            

            # Adding records from this epoch to files
            for idx in range(len(self.inference_samples)):
                if 'comet' in batch:
                    fp_comet = os.path.join(dir_infer, f"example_comet_{idx:03d}.csv")
                    
                    # comet- If file for example idx does not exists we add the true observed records
                    if not os.path.exists(fp_comet):
                        
                        df_comet = pd.DataFrame(columns=[ 'epoch', 'head','rels','tail', 'preds'])                    
                        
                        head = li_comet_heads[idx]
                        rels = li_comet_rels[idx]
                        tail = li_comet_tails[idx]
                        preds = li_comet_preds[idx]
                                            
                        datum = { 'epoch': 0,
                                    'head': head,
                                    "rels": rels,
                                    "tail":tail,
                                    "preds":preds }
                    
                        df_comet = df_comet.append(datum, ignore_index=True)
                        df_comet.to_csv( fp_comet, index=False)

                    # comet - adding to preds
                    df_comet = pd.read_csv(fp_comet)    
                    datum_comet = {
                        'epoch':df_comet['epoch'].max()+1,
                        'head': '',
                        'rels':'',
                        'tail':'',
                        'preds':li_comet_preds[idx] }

                    df_comet = df_comet.append(datum_comet, ignore_index=True)
                    df_comet.to_csv( fp_comet, index=False)

                if 'rst' in batch:
                    fp_rst = os.path.join(dir_infer, f"example_rst_{idx:03d}.csv")
                    # rst - If file for example idx does not exists we add the true observed records
                    if not os.path.exists(fp_rst):
                        
                        df_rst = pd.DataFrame(columns=[ 'epoch', 'head','rels','tail', 'preds'])                    
                        
                        head = li_rst_heads[idx]
                        rels = li_rst_rels[idx]
                        tail = li_rst_tails[idx]
                        preds = li_rst_preds[idx]
                                            
                        datum = { 'epoch': 0,
                                    'head': head,
                                    "rels": rels,
                                    "tail":tail,
                                    "preds":preds }
                    
                        df_rst = df_rst.append(datum, ignore_index=True)
                        df_rst.to_csv( fp_rst, index=False)

                    # rst - adding to preds
                    df_rst = pd.read_csv(fp_rst)    
                    datum_rst = {
                        'epoch':df_rst['epoch'].max()+1,
                        'head': '',
                        'rels':'',
                        'tail':'',
                        'preds':li_rst_preds[idx] }

                    df_rst = df_rst.append(datum_rst, ignore_index=True)
                    df_rst.to_csv( fp_rst, index=False)

    def create_data_loaders(self, shuffle=False, **kwargs ):
        
        dg = DataLoaderGenerator(self.dir_data_rst, self.dir_data_atomic2020,
                self.batch_size, self.model.pad_values,
                self.model.tokenizer, 
                workers=self.workers, mode=self.mode, split=self.data_splits,
                #dset_blend=self.dset_blend
                )

        
        if "train" in self.mode:
            self.train_dl = dg.prepare_dataloader_combined(True, 'train', loss_weight_rst=self.loss_weight_rst, loss_weight_comet=self.loss_weight_comet)
            self.val_dl = dg.prepare_dataloader_combined(True, 'val', loss_weight_rst=self.loss_weight_rst, loss_weight_comet=self.loss_weight_comet)
            self.test_dl = dg.prepare_dataloader_combined(False, 'test', loss_weight_rst=self.loss_weight_rst, loss_weight_comet=self.loss_weight_comet)
            self.inference_dl = dg.prepare_dataloader_combined(True, 'inference', loss_weight_rst=self.loss_weight_rst, loss_weight_comet=self.loss_weight_comet)        
        elif self.mode == "test":
            self.test_dl = dg.prepare_dataloader_combined(shuffle=True, split_name='test')

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    @lru_cache()
    def total_steps(self):

        ds_size = max( [ len(dl) for dl in self.train_dl.values()] ) // max(1,self.gpus)
        steps = (ds_size * self.max_epochs) // (self.accumulate_grad_batches)
        return steps

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
                
        warmup_steps = int( self.warmup_proportion*self.total_steps() )

        lr_schedule = get_cosine_schedule_with_warmup(optimizer, 
                        warmup_steps, self.total_steps(), 0.5 )

        return [optimizer], [{ "scheduler":lr_schedule ,"interval": "step", "monitor":"val_loss"}]
    
    def return_params(self):
        params = {}
        keys = ['batch_size','accumulate_grad_batches','learning_rate','max_epochs',
            'dir_data_rst','dir_data_atomic2020',
            'warmup_proportion','tag','version','loss_weight_rst','loss_weight_comet',
            'randomize_comet_pronouns','remove_to']
        
        params = {
            k:self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }

        #json_keys = ['data_splits','dset_blend']

        json_keys = ['data_splits']
        
        jparams = {
            k:ujson.dumps(self.__dict__[k]) for k in json_keys if k in self.__dict__.keys()
        }

        params = {**params, **jparams}

        return params


if __name__ == "__main__":

    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False) 
    
    tparams = TrainingModule.parse_train_specific_args(parent_parser)
    mparams = CombinedModel.parse_model_specific_args(parent_parser)

    if tparams.mode in [ "test","inference"]:
        assert tparams.gpus in [0,1]

    # adjust to allow ddp to work on computer
    if tparams.gpus not in [0,1]:
        os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'
        os.environ['MASTER_PORT'] = '65302'

    main(vars(tparams), vars(mparams))
    combined_model = CombinedModel( )