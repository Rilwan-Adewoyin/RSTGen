import train_comerst
import utils_comerst as utils
from train_comerst import TrainingModule
import torch

def main():
    model_name="COMERST"
    model_version = 2

    model = TrainingModule.load_comerst(model_name=model_name ,model_version=model_version)

    bad_words = [r"\n" ] 
    bad_words_ids = [model.tokenizer.base_tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words]
    bad_words_ids = bad_words_ids 

    generation_kwargs = {'num_beams':1, 'temperature':1.2, 'repitition_penalty':1.0, 
                        'early_stopping':False, 'do_sample':False, 'no_repeat_ngram_size':3, 
                        'bad_words_ids':bad_words_ids, 'num_return_sequences':1,
                        'min_length':2, 'max_length':20 } #'max_length':30,'min_length':4


    # Load test dataset
    fp = "./dataset_atomic2020/test_v2.csv"
    model.tokenizer.randomize_comet_pronouns=False 
    dset = train_comerst.SingleDataset_atomic2020( fp, model.tokenizer,drop_duplicates = True)
    batch_size =1
    assert batch_size == 1 
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False,
                                            num_workers=5,
                                        collate_fn=lambda batch: utils.default_collate_pad( batch, model.pad_values) )

    blue1_scores = []
    blue2_scores = []
    blue3_scores = []
    blue4_scores = []
    meteor_scores = []
    rouguel_scores = []
    CIDEr_scores = []
    BERT_Scores = []

    li_refs = []
    li_preds = []

    #TODO: stride through data

    # For Blue, METEOr, Rogue
    for idx, batch in enumerate( dloader ):
        head, rel, tail = dloader.dataset.data.iloc[idx]
        preds = model.generate_from_dloaderbatch( batch, comet_or_rst="comet", generation_kwargs=generation_kwargs )
        
        li_preds.extend(preds)
        li_refs.append(tail)

        print("\n")
        print("Head:",head,"\t\tRel:", rel, "\tTail",tail)
        print(preds[0])
        
    li_preds = [pred.strip() for pred in li_preds]
    
    blue1_scores = utils.calculate_bleu_score( li_preds, li_refs  )
    rouge_scores = utils.calculate_rouge(li_preds, li_refs  )



if __name__ == '__main__':
    main()

#TODO:
    # Write up motivation between RST and common sense kg connection 
    
    # Main idea: Consider mappings from RST rels to Common Sense relations
    
    # I1: test comerst model w/ mapping CSKG relations to RSTs
    # I2: First retreive common sense knowledge, then use that to predict the next keyphrase
        # map rst relation -> comet2020 key phrases
        # generate tail common sense knowledge

