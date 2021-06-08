from nltk.tree import Tree
import train_comerst
import train_nlg
import train_rstplanner

class Combined_model():

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

        self.nlg_model = train_nlg.load_nlgmodel(model_name="NLG_rt", model_version=11,max_input_len=None)
        self.comerst = train_comerst.load_comerst('COMERST', model_version=1)
        self.train_rstplanner = train_rstplanner.RSTPlanner()

    
    def generate(self):
        pass
    
    def sample_rst(self):

        rst_chain = self.train_rstplanner.sample_rst_chain( )

        rst_chain_decoded = self.train_rstplanner.deserialize_chain( )
    
    