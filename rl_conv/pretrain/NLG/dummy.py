import train_comerst
import utils_comerst as utils
from train_comerst import TrainingModule
import torch

def main():
    model_name="COMERST"
    model_version = 1

    model = TrainingModule.load_comerst(model_name=model_name ,model_version=model_version)

    bad_words = [r"\n" ] 
    bad_words_ids = [model.tokenizer.base_tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words]
    bad_words_ids = bad_words_ids 
    
    generation_kwargs = {'num_beams':1, 'temperature':1.2, 'repitition_penalty':1.0, 
                        'early_stopping':False, 'do_sample':False, 'no_repeat_ngram_size':3, 
                        'bad_words_ids':bad_words_ids, 'num_return_sequences':1 } #'max_length':30,'min_length':4


    # Load test dataset
    fp = "./dataset_atomic2020/test_v2.csv"
    model.tokenizer.randomize_comet_pronouns=False 
    dset = train_comerst.SingleDataset_atomic2020( fp, model.tokenizer)
    dloader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False,
                                            num_workers=1,
                                        collate_fn=lambda batch: utils.default_collate_pad( batch, model.pad_values) )

    # For Blue, METEOr, Rogue
    for idx, batch in enumerate( dloader ):
        head, rel, tail = dloader.dataset.data.iloc[idx]
        li_pred_text, li_refs = model.generate_from_dloaderbatch( batch, comet_or_rst="comet", generation_kwargs=generation_kwargs )
        print("head:",head,"\trel:", rel, "\ttail",tail)
        print(li_pred_text)
        print("\n")


if __name__ == '__main__':
    main()

