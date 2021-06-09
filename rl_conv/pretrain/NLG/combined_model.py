from nltk.tree import Tree
from torch.nn.modules.loss import CosineEmbeddingLoss
import train_comerst
import train_nlg
import train_rstplanner
import argparse
import ujson
class CombinedModel():

    # Generation method
        # Input Factors
            # 1)RST Tree:kwargs used by rst_planner
            # 2)Key_phrase: key phrases and (possibly) their position and (possibly) their Nuclearity
            # 3)NLG:
        # Prediction Order
            # All of RST Tree
            # All key phrases
            # Sequentially predict text, using previous text as word context and rst tree and key phrases to guide


    def __init__(self):
        
        rst_params = "{ \"sampling_method\":\"random\", \"cond_subreddit\":\"aggregated\", \"cond_rstlength\":7, \"reduce_rel_space\":True }"

        self.rstplanner = train_rstplanner.RSTPlanner()
        self.comerst = train_comerst.load_comerst('COMERST', model_version=1)
        self.nlg_model = train_nlg.load_nlgmodel(model_name="NLG_rt", model_version=11,max_input_len=None)
        
        
    def generate(self):
        
        # sample an rst template
        rst_template = self.sample_rst() 

        # sample key phrases 
        key_phrases = self.sample_keyphrases()

        # generate text
        generated_text = self.gen_text()
    
    def sample_rst(self):

        rst_chain = self.train_rstplanner.sample_rst_chain( )

        rst_chain_decoded = self.train_rstplanner.deserialize_chain( )
    
    def sample_key_phrase(self, )
        pass

    def gen_text(self, ):
        pass


    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
                
        parser.add_argument('--base_model_name', default='bart_base', required=False)

        parser.add_argument('--model_name', default='COMERST', required=False)
        
        #TODO: this is not implemented yet - the maximum lengths
        parser.add_argument('-mlh','--max_len_head', type= int, default=20 )
        parser.add_argument('-mlt','--max_len_tail', type= int, default=20 )
        parser.add_argument('-ments','--max_edu_nodes_to_select', type=int, default=-1, )
        parser.add_argument('-far','--filter_atomic_rels', type=lambda x: bool(int(x)), default=False, )
        parser.add_argument('-isv','--init_std_var', type=float, default=0.005)
        parser.add_argument('-dem','--dict_embed_mnorms', type=lambda x: ujson.decode(x), default={})
        parser.add_argument('-re', '--relation_embedding', type=str, choices=['flattened','hierarchical1','hierarchical2'], default='flattened' )
        parser.add_argument('-at','--attention_type',type=int, choices=[1,2], default=1)
        
        parser.add_argument('-sgbf','--scale_grad_by_freq', type=lambda x: bool(int(x)) , default=True, 
                help="Inverse the gradients to the emebdding layers based on the occurence of each index in the minibatch ")
        mparams = parser.parse_known_args( )[0]
       
        return mparams




    
if __name__ == "__main__":
    
    combined_model = CombinedModel( )