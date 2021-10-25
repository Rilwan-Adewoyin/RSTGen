import difflib
import editdistance
import math
import numpy as np
import re
import spacy
import string
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, BertForMaskedLM,  AlbertTokenizerFast, AlbertForPreTraining
from transformers import glue_convert_examples_to_features
from transformers.data.processors.utils import InputExample
from wmd import WMD
from torch.nn import CrossEntropyLoss
from spacy.language import Language


@Language.factory('spacy-similarity')
def spacy_similarity(nlp, name):
    return WMD.SpacySimilarityHook(nlp)

class GreunEval():

    def __init__(self,
        device = 'cuda:0',

        gmr_lm_model_name = None,
        gmr_cola_model_name = None,
        cohrnc_model_name = None
        ):
        
        # General Settings
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Grammaticality LM
        self.gmr_lm_model_name = gmr_lm_model_name if gmr_lm_model_name else 'bert-base-cased'
        self.gmr_lm_model = BertForMaskedLM.from_pretrained(self.gmr_lm_model_name)
        self.gmr_lm_model.eval()
        self.gmr_lm_tokenizer = BertTokenizer.from_pretrained(self.gmr_lm_model_name)

        #Grammarticality COLA
        self.gmr_cola_model_name = gmr_cola_model_name if gmr_cola_model_name else 'bert-base-cased'
        saved_pretrained_CoLA_model_dir = './models/cola_model/' + self.gmr_cola_model_name + '/'
        config_class, model_class, tokenizer_class = (BertConfig, BertForSequenceClassification, BertTokenizer)
        self.gmr_cola_config = config_class.from_pretrained(saved_pretrained_CoLA_model_dir, num_labels=2, finetuning_task='CoLA')
        self.gmr_cola_tokenizer = tokenizer_class.from_pretrained(saved_pretrained_CoLA_model_dir, do_lower_case=0)
        self.gmr_cola_model = model_class.from_pretrained(saved_pretrained_CoLA_model_dir, from_tf=bool('.ckpt' in self.gmr_cola_model_name), config=self.gmr_cola_config)
        self.gmr_cola_model.eval()

        #Focus Score
        nlp = spacy.load('en_core_web_md')
        # nlp.add_pipe(WMD.SpacySimilarityHook(nlp), last=True)
        nlp.add_pipe('spacy-similarity', last=True)
        self.focus_nlp = nlp
        

        #Coherence Score
        self.cohrnc_model_name = cohrnc_model_name if cohrnc_model_name else 'albert-base-v2'
        self.cohrnc_tokenizer = AlbertTokenizerFast.from_pretrained(self.cohrnc_model_name)
        self.cohrnc_tokenizer_params = {'max_length': 512,
                            'padding':'longest',
                            'truncation': 'only_second',
                            'return_tensors':'pt',
                            }
        self.cohrnc_model = AlbertForPreTraining.from_pretrained(self.cohrnc_model_name) #.to(device)
        self.cohrnc_loss_fct = CrossEntropyLoss()

    def to(self, device):

        self.gmr_lm_model.to(device)
        self.gmr_cola_model.to(device)
        self.cohrnc_model.to(device)

        self.device = device
        
    #region Processing
    """ Processing """
    def preprocess_candidates(self, candidates):
        for i in range(len(candidates)):
            candidates[i] = candidates[i].strip()
            candidates[i] = '. '.join(candidates[i].split('\n\n'))
            candidates[i] = '. '.join(candidates[i].split('\n'))
            candidates[i] = '.'.join(candidates[i].split('..'))
            candidates[i] = '. '.join(candidates[i].split('.'))
            candidates[i] = '. '.join(candidates[i].split('. . '))
            candidates[i] = '. '.join(candidates[i].split('.  . '))
            while len(candidates[i].split('  ')) > 1:
                candidates[i] = ' '.join(candidates[i].split('  '))
            myre = re.search(r'(\d+)\. (\d+)', candidates[i])
            while myre:
                candidates[i] = 'UNK'.join(candidates[i].split(myre.group()))
                myre = re.search(r'(\d+)\. (\d+)', candidates[i])
            candidates[i] = candidates[i].strip()
        processed_candidates = []
        for candidate_i in candidates:
            sentences = sent_tokenize(candidate_i)
            out_i = []
            for sentence_i in sentences:
                if len(sentence_i.translate(str.maketrans('', '', string.punctuation)).split()) > 1:  # More than one word.
                    out_i.append(sentence_i)
            processed_candidates.append(out_i)
        return processed_candidates
    #endregion

    #region Grammar Score
    def get_grammaticality_score(self, processed_candidates):
        lm_score = self.get_lm_score(processed_candidates)
        cola_score = self.get_cola_score(processed_candidates)
        grammaticality_score = [1.0 * math.exp(-0.5*x) + 1.0 * y for x, y in zip(lm_score, cola_score)]
        grammaticality_score = [max(0, x / 8.0 + 0.5) for x in grammaticality_score]  # re-scale
        return grammaticality_score

    """ Scores Calculation """
    def get_lm_score(self,sentences):

        lm_score = []
        for sentence in tqdm(sentences):
            if len(sentence) == 0:
                lm_score.append(0.0)
                continue
            score_i = 0.0
            for x in sentence:
                score_i += self.score_sentence(x, self.gmr_lm_tokenizer, self.gmr_lm_model)
            score_i /= len(sentence)
            lm_score.append(score_i)
        return lm_score

    def score_sentence(self, sentence, tokenizer, model):
        # if len(sentence.strip().split()) <= 1:
        #     return 10000
        tokenize_input = tokenizer.tokenize(sentence)
        if len(tokenize_input) > 510:
            tokenize_input = tokenize_input[:510]
        input_ids = torch.tensor(tokenizer.encode(tokenize_input)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            loss = model(input_ids, labels=input_ids)[0]
        return math.exp(loss.item())

    def get_cola_score(self, sentences):

        tokenizer, model = self.gmr_cola_tokenizer, self.gmr_cola_model

        candidates = [y for x in sentences for y in x]
        sent_length = [len(x) for x in sentences]
        cola_score = self.evaluate_cola(model, candidates, tokenizer, self.gmr_cola_model_name)
        cola_score = self.convert_sentence_score_to_paragraph_score(cola_score, sent_length)
        return cola_score


    def evaluate_cola(self, model, candidates, tokenizer, model_name):

        eval_dataset = self.load_and_cache_examples(candidates, tokenizer)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, sampler=torch.utils.data.SequentialSampler(eval_dataset), batch_size=max(1, torch.cuda.device_count()))
        preds = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                if model_name.split('-')[0] != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if model_name.split('-')[0] in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        return preds[:, 1].tolist()

    def load_and_cache_examples(self, candidates, tokenizer):
        max_length = 128
        examples = [InputExample(guid=str(i), text_a=x) for i,x in enumerate(candidates)]
        features = glue_convert_examples_to_features(examples, tokenizer, label_list=["0", "1"], max_length=max_length, output_mode="classification")
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_labels = torch.tensor([0 for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([[0.0]*max_length for f in features], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    def convert_sentence_score_to_paragraph_score(self, sentence_score, sent_length):
        paragraph_score = []
        pointer = 0
        for i in sent_length:
            if i == 0:
                paragraph_score.append(0.0)
                continue
            temp_a = sentence_score[pointer:pointer + i]
            paragraph_score.append(sum(temp_a) / len(temp_a))
            pointer += i
        return paragraph_score

    # endregion

    #region redundancy score
    def get_redundancy_score(self, all_summary):


        redundancy_score = [0.0 for x in range(len(all_summary))]
        for i in range(len(all_summary)):
            flag = 0
            summary = all_summary[i]
            if len(summary) == 1:
                continue
            for j in range(len(summary) - 1):  # for pairwise redundancy
                for k in range(j + 1, len(summary)):
                    flag += self.if_two_sentence_redundant(summary[j].strip(), summary[k].strip())
            redundancy_score[i] += -0.1 * flag
        return redundancy_score

    def if_two_sentence_redundant(self, a, b):
        """ Determine whether there is redundancy between two sentences. """
        if a == b:
            return 4
        if (a in b) or (b in a):
            return 4
        flag_num = 0
        a_split = a.split()
        b_split = b.split()
        if max(len(a_split), len(b_split)) >= 5:
            longest_common_substring = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
            LCS_string_length = longest_common_substring.size
            if LCS_string_length > 0.8 * min(len(a), len(b)):
                flag_num += 1
            LCS_word_length = len(a[longest_common_substring[0]: (longest_common_substring[0]+LCS_string_length)].strip().split())
            if LCS_word_length > 0.8 * min(len(a_split), len(b_split)):
                flag_num += 1
            edit_distance = editdistance.eval(a, b)
            if edit_distance < 0.6 * max(len(a), len(b)):  # Number of modifications from the longer sentence is too small.
                flag_num += 1
            number_of_common_word = len([x for x in a_split if x in b_split])
            if number_of_common_word > 0.8 * min(len(a_split), len(b_split)):
                flag_num += 1
        return flag_num

    def get_focus_score(self, all_summary):

        all_score = self.compute_sentence_similarity(all_summary)
        focus_score = [0.0 for x in range(len(all_summary))]
        for i in range(len(all_score)):
            if len(all_score[i]) == 0:
                continue
            if min(all_score[i]) < 0.05:
                focus_score[i] -= 0.1
        return focus_score

    def compute_sentence_similarity(self, all_summary):
        
        all_score = []
        for i in range(len(all_summary)):
            if len(all_summary[i]) == 1:
                all_score.append([1.0])
                continue
            score = []
            for j in range(1, len(all_summary[i])):
                doc1 = self.focus_nlp(all_summary[i][j-1])
                doc2 = self.focus_nlp(all_summary[i][j])
                try:
                    score.append(1.0/(1.0 + math.exp(-doc1.similarity(doc2)+7)))
                except:
                    score.append(1.0)
            all_score.append(score)
        return all_score
    #endregion

    def get_coherence_score(self, processed_candidates):
        #Calculate Inter Sentence coherence score
        #Extracting examples
        candidates_examples_pos = [None]*len(processed_candidates)
        candidates_examples_neg = [None]*len(processed_candidates)
        candidates_labels_pos = [None]*len(processed_candidates)
        candidates_labels_neg = [None]*len(processed_candidates)
        #Extract all possible consecutive pairs of segments
        for idx, candidate in enumerate(processed_candidates):
            positive_examples = list(zip( candidate, candidate[1:]))
            negative_examples = [ (seg_pair[1],seg_pair[0]) for seg_pair in positive_examples ]

            candidates_examples_pos[idx]=positive_examples
            candidates_examples_neg[idx]=negative_examples

            candidates_labels_pos[idx] = [1.0]*len(positive_examples)
            candidates_labels_neg[idx] = [0.0]*len(negative_examples)

        #Moving to torch
        #TODO: remember to cap each subsequence length to tokeinzer max_length of 512

        candidates_examples_encoded = [ self.cohrnc_tokenizer( li_pos+li_neg, **self.cohrnc_tokenizer_params) 
                            for li_pos,li_neg in zip( candidates_examples_pos, candidates_examples_neg) ]
        labels =   [ torch.tensor(labels_pos+labels_neg,dtype=torch.long) for labels_pos, labels_neg in 
                                zip(candidates_labels_pos, candidates_labels_neg) ]
                
        #For each candidate :
            # calculate the logistic losses non aggregated
        log_losses = [None]*len(processed_candidates)

        with torch.no_grad():

            for idx in range(len(processed_candidates) ):
        
                output = self.cohrnc_model(**candidates_examples_encoded[idx].to(self.device)) 
                log_loss = self.cohrnc_loss_fct( output.sop_logits.view(-1, output.sop_logits.size(-1)), labels[idx].to(self.device).view(-1) )
                log_losses[idx] = log_loss


        #Average the SOPlosses: avgsop
        final_coherence_scores = [ -l.cpu().numpy().tolist() for l in log_losses]
        return final_coherence_scores

    def get_greun(self, candidates):
        processed_candidates = self.preprocess_candidates(candidates)
        grammaticality_score = self.get_grammaticality_score(processed_candidates)
        redundancy_score = self.get_redundancy_score(processed_candidates)
        focus_score = self.get_focus_score(processed_candidates)
        coherence_score = self.get_coherence_score(processed_candidates)
        greun_score = [min(1, max(0, sum(i))) for i in zip(grammaticality_score, redundancy_score, focus_score, coherence_score)]
        return greun_score, grammaticality_score, redundancy_score, focus_score, coherence_score

    def __call__(self, candidates):
        return self.get_greun(candidates)

if __name__ == "__main__":
    candidates = [
        # "This is a good example. A good example.",
                
                "I fly boats between countries. A kangaroo lives in a zoo with elephants. What is the meaning of life?",
                  "This is a bad example. It is ungrammatical and redundant. Orellana shown red card for throwing grass at Sergio Busquets. Orellana shown red card for throwing grass at Sergio Busquets."
                  ,

                  "He just got the circuit to turn the two leads into the floor. He was on the breakers.\
                       A GFCI is not going to be able to do that without the high voltage lead. It's not a single prong,\
                            it's his resistance. You can't just throw water at him. That's why you can get water if you're\
                                 not using outdoor outlets or bathroom",
                  

                  "President Clinton has done more harm to the respect of the Presidency than Presidents Presidents Ford, \
                    Carter and Reagan have demonstrated, and that the failure of Presidents Clinton and Reagan has caused \
                     permanent damage to the office. You omit President Bush from your list of Presidents. Mr. Bush has brought honor,\
                  dignity and respect to the office of the Presidency in the eyes of the American people and the world at large.\
                       We owe your readers an apology and a correction of your editorials glaring omission, which created an erroneous impression"
                    
                #     ,
                # "Climate change denial is driven by fossil fuel industries that are paralyzing the humanities in attempting to deal\
                # with the problem. The death toll from ensuing CO2 is staggering. There is no winning consequence for anything \
                # remotely dire that climate change denialism has. Leaving aside the issue of GM risk, simply comparing climate \
                # change denialism to climate change protesters is not going to stop anything."
                  ]
    
    greun_evaluator = GreunEval()
    greun_evaluator.to(greun_evaluator.device)
    greun_score, grammaticality_score, redundancy_score, focus_score, coherence_score = greun_evaluator(candidates)
    greun_evaluator.to('cpu')
    print(greun_score)