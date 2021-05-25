# Extracted from https://github.com/allenai/comet-atomic-2020/blob/master/models/comet_atomic2020_bart/utils.py
  
import itertools
import json
import linecache
import os
import pickle
import warnings
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List

dirname = os.path.dirname(__file__)
from copy import deepcopy

import git
import numpy as np
import torch
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
#from transformers.models.bart import BaseModelOutput
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_convert
from torch._six import string_classes

from typing import Optional, Callable, Union, Optional, List, Iterable
import collections

import random 
import regex as re
np_str_obj_array_pattern = re.compile(r'[SaUO]')


huggingface_names = {'bart_base': "facebook/bart-base"}

def load_pretrained_transformer( model_name='bart', transformer=True, 
                                    tokenizer=False):
    _dir_transformer = os.path.join( get_path("./models"), model_name )
    exists = os.path.isdir(_dir_transformer)
    output = {}
    
    if exists == False:    
        model_tokenizer = AutoTokenizer.from_pretrained(huggingface_names[model_name])
        model = BartForConditionalGeneration.from_pretrained(huggingface_names[model_name] )
        
        model_tokenizer.save_pretrained(_dir_transformer)
        model.save_pretrained(_dir_transformer)

    if tokenizer == True:
        output['tokenizer'] = AutoTokenizer.from_pretrained(_dir_transformer)

    if transformer == True:
        output['transformer'] = BartForConditionalGeneration.from_pretrained(_dir_transformer)
    
    return output

def load_base_tokenizer( model_name,
                            dir_tokenizer,
                            base_tokenizer_name,
                            output_version="phrase"
                            ):
    
    if os.path.isdir(dir_tokenizer):
        base_tokenizer = AutoTokenizer.from_pretrained(dir_tokenizer, use_fast=False)

    # retreiving base tokenizer from online or from local distillgpt2
    else:
        dir_transformer = os.path.join("./models", base_tokenizer_name)
        exists = os.path.isdir(dir_transformer)            

        if exists==True:
            base_tokenizer = AutoTokenizer.from_pretrained(dir_transformer, use_fast=False)
            config = AutoConfig.from_pretrained(dir_transformer)

        elif exists==False:
            base_tokenizer = AutoTokenizer.from_pretrained(huggingface_names['base_tokenizer_name'],use_fast=False)
            config = AutoConfig.from_pretrained(huggingface_names['base_tokenizer_name'])

            os.makedirs(dir_tokenizer)
                
            base_tokenizer.init_kwargs['name_or_path'] = dir_tokenizer
                #base_tokenizer.init_kwargs['special_tokens_map_file'] = os.path.join(dir_tokenizer,"special_tokens_map.json")
                
            base_tokenizer.save_pretrained(dir_tokenizer)
            config.save_pretrained(dir_tokenizer)
    
    return base_tokenizer

def BART_encoder_forward(
    self,
    input_ids=None,
    attention_mask=None,
    head_mask=None,
    inputs_embeds=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    ):
    r"""
    MONKEY_PATCHED: Allowed for position_ids and position_embeds to be passed in. Before the position ids were auto generated

    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
            provide it.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
            for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the heas is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
            into associated vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
            returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
            for more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

    # HACK
    #embed_pos = self.embed_positions(input_shape)
                                # HACK
    hidden_states = inputs_embeds #+ embed_pos
    hidden_states = self.layernorm_embedding(hidden_states)
    hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

    # expand attention_mask
    if attention_mask is not None:
        
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask_2(attention_mask, inputs_embeds.dtype)
        
    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (
            len(self.layers)
        ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):  # skip the layer
            layer_outputs = (None, None)
        else:
            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
    )

def BART_forward(
    #region All the same
    self,
    input_ids=None,
    attention_mask=None,
    decoder_input_ids=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    encoder_outputs=None,
    past_key_values=None,
    inputs_embeds=None,
    decoder_inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    ):

    # different to other models, Bart automatically creates decoder_input_ids from
    # input_ids if no decoder_input_ids are provided
    if decoder_input_ids is None and decoder_inputs_embeds is None:
        decoder_input_ids = shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if encoder_outputs is None:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )
    #endregion
    # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
    # NOTE : change here
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_outputs[0],
                #NOTE: Akanni Hack. 1st adapt mask for decoder, then expand it for your code
            #    we feed a non generic 3 dim (bs, tgt_len, src_len ) attn_mask to encoder.
            #   IN source code The decoder uses the 2dim encoded_attn_mask to decide where to attnd
            #   Our 3dim mask is not interoperable with this system.
            #   So we reduce our 3dim mask to a 2dim mask (bs, src_len) after it has been used by the encoded
            #   This 2dim encoded_attn mask will be 1 for every non pad position and 0 otherwise
                # This is achieved by putting 1 for every src_len position that has at least one other position attending to it
                    # Since this indicates it is not a mask token
        encoder_attention_mask=torch.where( attention_mask.sum(dim=-1)==0,0,1),
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
        inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if not return_dict:
        return decoder_outputs + encoder_outputs

    return Seq2SeqModelOutput(
        last_hidden_state=decoder_outputs.last_hidden_state,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


def default_collate_pad(batch, pad_values=None):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)


    if isinstance(elem, torch.Tensor):
                   
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)

        return torch.stack(batch, 0, out=out)


    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate_pad([torch.as_tensor(b) for b in batch], pad_values)

        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        dict_output = {}
        for key in elem:
            li_ = [d[key] for d in batch]

            #it = iter(batch)
            elem_size = len(li_[0])

            if not all(len(elem_) == elem_size for elem_ in li_):
                # raise RuntimeError('each element in list of batch should be of equal size')
                # it = iter(batch)
                largest_seq = max( len(elem_) for elem_ in li_ ) 
                
                #handling 2d attention mask
                if li_[0].dim() == 2:
                    
                    for idx in range(len(li_)):
                        elem_ = li_[idx]
                        missing_dims = largest_seq - len(elem_)
                        if missing_dims > 0:
                            # adding missing_dims paddings to dim 1 which reflects masking the new padding tokens
                            # adding paddings value 0 - to dim 0 which reflects the 

                            elem_ = torch.nn.functional.pad( elem_, (0, missing_dims, 0, missing_dims), 
                                mode='constant', value=0.0 )
                            li_[idx] = elem_
                            
                elif li_[0].dim() == 1:
                    padded_li = pad_sequence(li_, batch_first=True, padding_value=pad_values[key] ) 
                    #unstacking
                    li_ = torch.unbind(padded_li, 0)

            dict_output[key] = default_collate_pad( li_, pad_values )    

        #dict_output =  {key: default_collate_pad([d[key] for d in batch], pad_values) for key in elem }
        return dict_output

    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_pad(samples,pad_values) for samples in zip(*batch)))

    raise TypeError(default_collate_err_msg_format.format(elem_type))
             
def get_path(_path,_dir=False):

    if os.path.isabs(_path) == False:
        _path = os.path.join(dirname, _path)
    
    _path = os.path.realpath(_path)
    
    if _dir:
        os.makedirs(_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(_path), exist_ok=True)

    return _path

def _expand_mask_2(mask, dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, tgt_len, src_len = mask.size()

    expanded_mask = mask[:, None, :, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

def monkey_save_model(self, trainer, filepath: str):
    # in debugging, track when we save checkpoints
    trainer.dev_debugger.track_checkpointing_history(filepath)

    # make paths
    if trainer.is_global_zero:
        self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

    # delegate the saving to the trainer
    if self.save_function is not None:
        self.save_function(filepath, self.save_weights_only)
    
    self.to_yaml()

def _monitor_candidates(self, trainer):
    ckpt_name_metrics = deepcopy(trainer.logger_connector.logged_metrics)
    ckpt_name_metrics.update(trainer.logger_connector.progress_bar_metrics)
    ckpt_name_metrics.update(trainer.logger_connector.callback_metrics)

    return ckpt_name_metrics



########### 0---------------
def encode_line(tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu_score(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": corpus_bleu(output_lns, [refs_lns], **kwargs).score}



class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id) -> tuple:
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
        }
        return batch

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.src_lens, batch_size)


class MBartDataset(Seq2SeqDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.max_source_length != self.max_target_length:
            warnings.warn(
                f"Mbart will ignore max_target_length = {self.max_target_length} and use {self.max_source_length} for both sides."
            )

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {
            "tgt_texts": source_line,
            "src_texts": tgt_line,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_translation_batch(
            [x["src_texts"] for x in batch],
            src_lang=self.src_lang,
            tgt_texts=[x["tgt_texts"] for x in batch],
            tgt_lang=self.tgt_lang,
            max_length=self.max_source_length,
        )
        return batch_encoding.data


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size):
        self.data, self.bs = data, batch_size

    def key(self, i):
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data))
        sz = self.bs * 50
        ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path):
    with open(path, "w") as f:
        json.dump(content, f, indent=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
    }
    return repo_infos


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def calculate_rouge(output_lns: List[str], reference_lns: List[str], use_stemmer=True) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}


def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


comet_tail_to_ignore1 = [ "none", "NONE" ,np.nan, '?',
        'l', '3', 't', 'Y', 'X', 'g', '0', 'F', 'q', 'v', '`', 'h', 's', 'a', 'n', 'c','1',
        'd', 'e', 'i', 'ok', 'no', 'xx', 'NO','na', 'aF', 'N/', 'to', 'sd', 'up', 'it',
        'Hi', 'tv', 'Na', 'me', 'be',
        'iv', 'cd', 'co', 'st', 'us', 'or', '4h',
        'oz', 'fl', 'in', 'rv','uk', 'do', 'mb', 'li', 'ai', 'g4', 'vd',
        'go', 'ex', 'c9', '21', 'el', '2h', 'ox', 'on',
        'q\\', 'ge', 'ru', 'th', 'TV', 'ID', 'Id', 'HR', 'sw', 'CD', 'ii']

comet_tail_to_ignore2 = [np.nan,
        'q', '`',
        '?', 'v', 'to', 'ok', 'NO', 'na', 'no',
        'go', 'tv', 'TV', 'do', 'ar', 're', 'it', 'PC', 'me']

comet_tail_to_ignore3 = [np.nan,
        'o', 'B', 'a', 'u',
        'r', 'Y', 'q', 'ok', 'na', 'NO', 'no', 'aC', 'to', 'in',
        'st', 'up', 'do', 'go', 'be', 'un', 'tv', 'TV', 'ID', 'ox', 'CD']

comet_tail_to_ignore = list( set( comet_tail_to_ignore1 + comet_tail_to_ignore2 + comet_tail_to_ignore3 ) )