import os

import numpy as np
import pandas as pd
import argparse
import glob
import json
import pickle
import copy 

import ujson
import inspect
import math

from collections import defaultdict, Counter, ChainMap
# nested default dict
def nested_dd():
    return defaultdict(nested_dd)

#TODO: write script v1
#TODO: add ability to condition on source dataset
#TODO: add ability to sample
#TODO: add ability to sample with conditions


class RSTPlanner():

    def __init__(self, model_name, dirs_rst_convo,
                    maximum_child_node_pos=30, **kwargs ):

        self.model_name = model_name
        self.dirs_rst_conv = dirs_rst_convo
        
        # dictionary to hold list of rst_sequences for each subreddit
            # we usually flush this of its content
        self.dict_subreddit_lirstseq = {}
        
        # dictionary to hold list of conditional counters
        self.dict_subreddit_dict_rstt_counter_rsttplus1 = {}
        
        self.start_rst_chunk = {'rel':'start', 'pos':-1, 'ns':'start'}
        self.edu_rst_chunk = {'rel':'edu', 'pos':-1, 'ns':'edu'} #this signifies an edu has been reached

        self.rng = np.random.default_rng()
        self.maximum_child_node_pos = maximum_child_node_pos
        self.li_subreddit_names = list( filter( lambda name: name!="last_batch_record", os.listdir( self.dirs_rst_conv ) ) )
        
        self.save_dir = "./RSTMarkov"
        os.makedirs( self.save_dir, exist_ok=True )

        filter_names = filter( lambda x: x not in ["self", "li_rstseq"] , inspect.getfullargspec(self.filter_rstseq).args)
        self.filter_names = sorted(filter_names)
    
        # Load in or create the record of existing markovian distributions made
        self.path_markov_distr_records = os.path.join( self.save_dir, "markov_distr_records.csv" )
        if os.path.exists( self.path_markov_distr_records ):
            self.df_markovdistr_records = pd.read_csv( self.path_markov_distr_records, index_col=0  )
            self.df_markovdistr_records = self.df_markovdistr_records.where(self.df_markovdistr_records.notnull(), 'null')

        else:
            columns = ['subreddit', 'version'] + self.filter_names + ['file_path']
            self.df_markovdistr_records = pd.DataFrame(columns=columns)
        
        # Storage for the subreddits and markov distribution
        self.dict_subreddit_markovdistr = {}
    
    def subreddit_name_in_records(self, subreddit):
        subreddit_name = copy.deepcopy( subreddit )
        if subreddit_name == "aggregated":
            subreddit_name = "_".join( sorted( self.dict_subreddit_markovdistr.keys() ) )
        return subreddit_name

    #@lru.cache
    def get_markovchain_savepath(self, subreddit, data_filters):
        """Creates a name for the markov chain we aim to generate

        Args:
            subreddit ([type]): [description]
            data_filters ([type]): [description]
        """
        
        subreddit_name = self.subreddit_name_in_records(subreddit)

        li_dict_records = self.df_markovdistr_records.to_dict('records')

        # subreddit filtering
        li_dict_records_subredditfilt = [ _dict for _dict in li_dict_records if _dict['subreddit']==subreddit_name ] 

        # creating a list of the filters already used for this subreddit
        li_dict_filtered_subreddit_filters = [ { k:v for k,v in _dict.items() if k in self.filter_names } for _dict in li_dict_records_subredditfilt ]
        
        # if filters already exist with a path in the records then just extract it
        data_filters_encoded = { k:json.dumps(v) for k,v in data_filters.items() }
        if data_filters_encoded in li_dict_filtered_subreddit_filters:
            index = li_dict_filtered_subreddit_filters.index( data_filters_encoded )
            path_markov_distribution = li_dict_records_subredditfilt[index]['file_path']
            version = li_dict_records_subredditfilt[index]['version']

        else:
            # making 
            # getting the version as the next available number in the integer encoding to filter combination
            existing_versions = [ _dict['version'] for _dict in li_dict_records_subredditfilt ]
            version = max(existing_versions, default=-1) + 1
            save_fn = f"{subreddit_name}_v{version:0>4}.pickle"
            os.makedirs( os.path.join(self.save_dir,"markov_distributions"), exist_ok=True )
            path_markov_distribution = os.path.join(self.save_dir,"markov_distributions", save_fn )

        return path_markov_distribution, subreddit, version

    def create_markovian_distr(self, subreddit, data_filters={} , smoothing_method=None):
        """Create or load a markov distribution for subreddit subreddit"""

        print(f"Creating Markov Chain for {subreddit}")

        # filling in missing filters that user may not have entered
        data_filters = { k:data_filters[k] if k in data_filters else None for k  in self.filter_names  }

        #filling in any non present filter
        #Creating/retreiving filepath
        save_fp, subreddit, version = self.get_markovchain_savepath(subreddit, data_filters)
        
        if save_fp in self.df_markovdistr_records.file_path.values.tolist():
            markovdistr = pickle.load( open(save_fp, "rb") )
            subreddit_name_in_records = self.subreddit_name_in_records(subreddit)
            self.dict_subreddit_markovdistr[subreddit] = markovdistr
        # Check if an existing record of a markov distribution with specific filters exists in the csv file
        else:
            
            if subreddit != "aggregated":
                self.collect_data(subreddit, data_filters)

            # create conditional counts of each rst
            self.create_transition_counts( subreddit )   
            # perform any smoothing
            if smoothing_method!=None:
                pass
            
            # create conditional  probabilistic distributions
            dict_rstt_condprobs_rsttplus1 = self.create_cond_probabilities( subreddit)
            self.dict_subreddit_markovdistr[subreddit] = dict_rstt_condprobs_rsttplus1
            # save file to record
            pickle.dump( dict_rstt_condprobs_rsttplus1, open(save_fp, "wb") )
            self.update_markov_distribution_records( subreddit, data_filters, save_fp, version )
               
    def collect_data(self, subreddit_name, data_filters={} ):
        #TODO: add filters on which data we collect later
        # get list of files
        ls_fp = glob.glob( os.path.join( self.dirs_rst_conv, subreddit_name, "*" ) ) #this may include a lock file generated by data_setup.py
        fp = [fn for fn in ls_fp if os.path.split(fn)[-1]!="lock"][0]
                
        fs =  int(fp[-10:]) 
            #defining starting line and total lines to use for dataset
        line_start = 0
        line_end = line_start + int(fs*0.6)

        with open(fp, 'r') as f:
            # load into memory only the rst chains
            data = pd.read_csv(f, sep=',', header=0, 
                nrows=(line_end-line_start), usecols=['rst'] )

            li_rstseq = [ ujson.loads(rstseq) for rstseq in  data['rst'].values.tolist() ] 
                # adding start and ending token rst encodings

            if len(data_filters) > 0 :

                # filtering sample dataset
                li_rstseq = self.filter_rstseq(li_rstseq, **data_filters )

        self.dict_subreddit_lirstseq[subreddit_name] = [ [self.start_rst_chunk] + rst_seq for rst_seq in li_rstseq ] #List of rst strings
    
    def filter_rstseq( self, 
                        li_rstseq,

                        li_RST_len=None, #filtering on rst length
                        
                        dict_RSTrel_counts=None, #filtering on a minimum, maximum, or exact amount of rsts in a sequence
                        rst_rel_count_mode=None, #'greaterequal_than',
                        
                        li_RSTchunk_to_include=None #[] #filtering on containing a specific RST chunks in the sequence 
                                                        # usually a user will place complete restraints such as [ {pos:"1", rel:"Elaboration","ns":NN} ]
                                                            # which means this will appear at this posiiotn
                         ):
        #TODO:  change this to be what you want 

        # RST_len filtering            
        if li_RST_len != None:
            li_rstseq =  [ rstseq for rstseq in li_rstseq if len(rstseq) in li_RST_len ]

        # RST relation count filtering
            # placing condiitons on the presence of specific counts relations in the sequences
        if dict_RSTrel_counts != None:
            assert rst_rel_count_mode in ["greaterequal_than", "equal_to", "lessequal_than"]
            def func_rstrelcountfilt(rst_seq ):
                rel_counter = Counter( [ dict_['rel'] for dict_ in rst_seq] ) 
                
                if rst_rel_count_mode == "greaterequal_than":
                    bool_ = any( [ rel_counter[rel] >= dict_RSTrel_counts[rel] for rel in dict_RSTrel_counts.keys() ] ) 
                
                elif rst_rel_count_mode == "equal_to":
                    bool_ = any( [ rel_counter[rel] == dict_RSTrel_counts[rel] for rel in dict_RSTrel_counts.keys() ] ) 
                
                elif rst_rel_count_mode == "lessequal_than":
                    bool_ = any( [ rel_counter[rel] <= dict_RSTrel_counts[rel] for rel in dict_RSTrel_counts.keys() ] ) 
                    
                return bool_
            
            li_rstseq = [ rstseq for rstseq in li_rstseq if func_rstrelcountfilt(rst_seq) ]
            
        
        # ensuring the presence of certain rstchunks in the li_RST
        if li_RSTchunk_to_include != None:

            def func_rstchunkfilt(rst_seq):
                
                bool_ = all( rstchunk in rst_seq for rstchunk in li_RSTchunk_to_include  )

                return bool_

            li_rstseq = [ rstseq for rstseq in li_rstseq if func_rstchunkfilt(rstseq) ]

        return li_rstseq

    def create_transition_counts(self, subreddit_name ):
        # create a record of transition counts between rst_chunks
        # if parent node pos n appears in rst_seq:
        # check for any child nodes appear in rst_seq:
            # for each child node pos cn that appears:
                # add cn to conditional counter for pos n
            
            # for each child node pos cn that does not appear:
                # add a count to the sef.end_rst_chunk
        assert subreddit_name in self.dict_subreddit_lirstseq or subreddit_name=="aggregated"
        
        if subreddit_name != "aggregated":
        
            li_rstseq = self.dict_subreddit_lirstseq[subreddit_name]
            
            dict_rstt_counter_rsttplus1 = defaultdict( Counter )  # Counter: key is rst_(t), val is counter for the distr of rst_(t+1)

            # iterating through li_rstseqs in one subredit dataset
            for rst_seq in li_rstseq:
                                
                li_rstchunknodepos = [ rstchunk['pos'] for rstchunk in rst_seq ]
                
                for rst_t in rst_seq:
                
                    # gathering pos of potential child nodes
                    if rst_t['pos'] == -1 :
                        li_child_node_pos = [ 0 ]
                    else:
                        # formula to find child nodes
                        li_child_node_pos = [ 2*rst_t['pos'] +1, 2*rst_t['pos'] + 2 ]
                    
                    for child_nodes_pos in li_child_node_pos:

                        # If child_node_pos appear in rst_seq and less than max child pos, add it to counter
                        if child_nodes_pos in li_rstchunknodepos and child_nodes_pos<self.maximum_child_node_pos:
                            _ = li_rstchunknodepos.index( child_nodes_pos )    
                            child_node_rstchunk = rst_seq[_]
                        
                        else:
                            # if it does not then add the edu marker token
                            child_node_rstchunk = self.edu_rst_chunk

                            # correcting the position of the edu chunk
                            child_node_rstchunk['pos'] = child_nodes_pos

                        # Adding to conditional counter
                        dict_rstt_counter_rsttplus1[ujson.dumps(rst_t)][ ujson.dumps( child_node_rstchunk ) ] += 1
        
        else:
            # create a conditional probably matrix for all subreddits combined using micro averging
            dict_rstt_counter_rsttplus1 = nested_dd()

                # Forming a list of lists. Each inner list is a list of possible values for rst_t
            li_rstt = [ list( dict_rstt_counter_rsttplus1 ) for dict_rstt_counter_rsttplus1 in self.dict_subreddit_dict_rstt_counter_rsttplus1.values()  ]
            li_rstt = sum( li_rstt, [])  #flatten
            li_rstt = list( set( li_rstt  ) ) #remove any repeats

            # for each conditioning rst value
            for rstt in li_rstt:

                # list of conditional prob dicts for a specific rstt. Across all subreddits 
                counters = [ Counter(dict_rstt_counter_rsttplus1[rstt]) for dict_rstt_counter_rsttplus1 in  self.dict_subreddit_dict_rstt_counter_rsttplus1.values()  ]
                
                # Aggregating the freq of rst_tplus1 given rst_t across all subreddits
                dict_rstt_counter_rsttplus1[rstt] =   dict( sum(counters, Counter() ) )

        self.dict_subreddit_dict_rstt_counter_rsttplus1[subreddit_name] = dict_rstt_counter_rsttplus1

    def smoothen_transition_counts(self, smoothing_method ):
        pass

    def create_cond_probabilities(self, subreddit ):
        
        dict_rstt_counter_rsttplus1 = self.dict_subreddit_dict_rstt_counter_rsttplus1[subreddit]
        dict_rstt_condprobs_rsttplus1 = nested_dd()

        # creating conditional probabilites 
        # for each json_rst_chunk at position t, nomralize conditional counter into probabilities
        for json_rst_chunk in dict_rstt_counter_rsttplus1:
            cond_counter = dict_rstt_counter_rsttplus1[json_rst_chunk]
            cond_probs =  self.renormalize_probs(cond_counter)
            dict_rstt_condprobs_rsttplus1[json_rst_chunk] = cond_probs
        
        return dict_rstt_condprobs_rsttplus1

    def renormalize_probs(self, cond_distr):
        """[summary]

        Args:
            cond_distr ([dict]): [key:rst_chunk, value:probability]

        Returns:
            [type]: [description]
        """
        total = sum(  cond_distr.values() )
        cond_distr = { k:v/total for k,v in cond_distr.items() }
        return cond_distr

    def update_markov_distribution_records(self, subreddit, data_filters, save_fp, version):
        
        data_filters = { k:json.dumps(v) for k,v in data_filters.items() }
        new_datum = pd.DataFrame.from_records([{** {'subreddit':subreddit, 'file_path':save_fp, 'version':version}, **data_filters } ] )
        self.df_markovdistr_records = self.df_markovdistr_records.append( new_datum, ignore_index=True )
        self.df_markovdistr_records.to_csv( self.path_markov_distr_records )

    def sample_rst_chain(self, cond_subreddit="aggregated", sampling_method="greedy", cond_rstlength=None, cond_parentnodes=[], 
                            cond_parentnodes_samplebetween=False, reduce_rel_space=False ):
        #TODO: change to relect idea we are no sampling down tree branches
        """[summary]

            Args:
                cond_rstlength ([int], optional): [Conditional number of RST parent nodes]. Defaults to None.
                cond_parentnodes ([dict], optional): [Conditional information about any parent node in the RST tree]. Defaults to [].
                    for example [ {'rel':'Elaboration' , 'pos':0, 'ns':NS }, {'rel':None , 'pos':1, 'ns':None }  ] 
                    to indicate parent node position 1 has no constraint on relation and ns

                cond_parentnodes_samplebetween (bool, optional): [Boolean indicating whether the parentnodes specified in cond_parentnodes,
                                                    should have other parent nodes placed inbetween them or not ]
        """
        # Assertion checks
        assert cond_rstlength==None or cond_rstlength >= len(cond_parentnodes) 

        if cond_rstlength != None:
            assert sampling_method != "greedy"

        if cond_subreddit not in self.dict_subreddit_markovdistr:
            self.create_markovian_distr(cond_subreddit, data_filters={} , smoothing_method=None)

        markov_distribution = self.dict_subreddit_markovdistr[cond_subreddit]

        rst_seq = copy.deepcopy( cond_parentnodes )
        rst_seq = [self.start_rst_chunk] + rst_seq # adding initial chunk

        li_sampled_node_pos = [rst_chunk['pos'] for rst_chunk in rst_seq] #to store the parent nodes pos in the rst_seq

        # for each position in the binary tree
        for _index in range(-1, self.maximum_child_node_pos):
            
            # if an rst_chunk exists at position i+1:
                # if cond_parentnodes_fillbetween == False:
                    # continue
                # else:
                    # define sample_restrictions = { 'pos':[list of possible positions], 'rel':[list of possible rels], 'ns':[list of possible nuclearities] }
            
            curr_rstchunk_pos = _index

            # checking if this nodes parent nodes exist in the list of sampled nodes that are not edu nodes. 
                # Otherwise it must be skipped
            if curr_rstchunk_pos != -1:
                position_of_parent = math.floor( ( curr_rstchunk_pos-1 ) / 2 )
                
                # parent exists check
                if position_of_parent not in li_sampled_node_pos:
                    continue
                # parent not edu check
                elif position_of_parent  in li_sampled_node_pos:
                    parent_rst_chunk =  rst_seq[ li_sampled_node_pos.index(position_of_parent) ]
                    if parent_rst_chunk['rel'] == self.edu_rst_chunk['rel']: continue

            # Checking if this node is an EDU in which case we will skip to next index position
            #if curr_rstchunk_pos in li_sampled_node_pos:
            if curr_rstchunk_pos in li_sampled_node_pos:
                curr_rstchunk  = rst_seq[li_sampled_node_pos.index(curr_rstchunk_pos)]
            else:
                break

            bool_edu = curr_rstchunk['rel'] == self.edu_rst_chunk['rel'] #edu chunk has a rel of edu
            if bool_edu:
                continue
        
            # Getting pos of child nodes
            if curr_rstchunk_pos == -1:
                positions_of_children = [ 0] 
            else:
                positions_of_children = [ 2*curr_rstchunk_pos + 1, 2*curr_rstchunk_pos + 2 ]
           
            # list of children nodes to sample. 
            
            # We sample for child positions that are not in the li_sampled_node_pos or  dont have fully defined rst chunks
            pos_of_children_to_samplefully = [ pos for pos in positions_of_children if pos not in li_sampled_node_pos ]
            pos_of_children_to_samplepartially = [ pos for pos in positions_of_children if (pos in  li_sampled_node_pos) and any( val==None for val in rst_seq[li_sampled_node_pos.index(pos)].values() ) ]
            
            pos_of_children_to_sample = pos_of_children_to_samplefully + pos_of_children_to_samplepartially

            # sampling children of curr_rstchunk
            for pos in pos_of_children_to_sample:

                # getting restrictions entered by user for this next sample
                if pos in  li_sampled_node_pos:
                    restrictions = { k:v for k,v in rst_seq[li_sampled_node_pos.index(pos)].items() if v is not None}
                else:
                    restrictions = {}

                # Adapting markov distr to ensure only a specific child node is being sampled
                    # example. given node 1, we are removing occurences of all nodes 2 from conditional markv distr for node 0
                markov_distribution_1 = copy.deepcopy( markov_distribution )
                markov_distribution_1[ ujson.dumps(curr_rstchunk) ]  = Counter( { k:v for k,v in  markov_distribution_1[ ujson.dumps(curr_rstchunk) ].items() if ujson.loads(k)['pos']==pos } )
                markov_distribution_1[ ujson.dumps(curr_rstchunk) ] = self.renormalize_probs( markov_distribution_1[ ujson.dumps(curr_rstchunk) ] )

                # # Adapting the markov distribution to ensure that cond_rstlength is satisfied
                # rst_seq_ex_edus = [ rst_chunk for rst_chunk in rst_seq if rst_chunk['rel'] not in ['start','edu'] ]
                rst_seq_ex_edus = [ rst_chunk for rst_chunk in rst_seq if rst_chunk['rel'] not in ['start','edu'] ]
                if cond_rstlength != None and len(rst_seq_ex_edus)>cond_rstlength and curr_rstchunk_pos != -1:
                    break
                # elif cond_rstlength != None and len(rst_seq_ex_edus)<cond_rstlength and curr_rstchunk_pos != -1:
                #     # adapts the conditional distribuiton of the curr_rstchunk distr
                #     markov_distribution_1 = self.sample_length_enforcer( markov_distribution_1, cond_rstlength, curr_rstchunk, rst_seq, pos )
                # else:
                #     markov_distribution_1 = markov_distribution_1                
                
                sampled_child_node = self.sample_from_markov_distr( rst_seq, cond_rstlength ,markov_distribution_1,
                                        curr_rstchunk, sampling_method, reduce_rel_space,
                                        **restrictions )
                               
                # for non fully defined rst_chunks, removing the non fully defined chunk from list
                if pos in li_sampled_node_pos:
                    _ = li_sampled_node_pos.index(pos)
                    li_sampled_node_pos.pop(_)
                    rst_seq.pop(_)

                # adding sampled node to tree
                rst_seq += [sampled_child_node]
                li_sampled_node_pos += [ sampled_child_node['pos'] ]
        
        # cleaning up rst chain by removing start and edu rst_chunks
        rst_seq = [rst_chunk for rst_chunk in rst_seq if rst_chunk['rel'] not in ['start','edu']  ]

        return rst_seq
     
    def sample_from_markov_distr(self, rst_seq, cond_rstlength,markov_distribution, rst_cond_chunk, sampling_method='greedy' , reduce_rel_space=False, rel=None, ns=None):
        #TODO: change to relect idea we are no sampling down tree branches

        """[summary]

        Args:
            markov_distribution ([dict]): [key=rst_chunk_A, value=dictionary where k=rst_chunk_B, v=p(rst_chunk_B|rst_chunk_A) ]
            rel_restrictions ([rel], optional): [list of relations allowed for conditional sample]
            ns_restrictions ( [ns], optional) : [list of ns allowed for conditional sample]
            sampling_method ( ) : "greedy" or "random"

        Returns:
            [type]: [description]
        """
        # extract specific conditional distribution for rst_t from the markov distribution
            # TODO: if it does not exist, think of way to smooth to create a new markov distr


        cond_distr = copy.deepcopy( markov_distribution[ ujson.dumps(rst_cond_chunk)] ) #cond_distr= { rst_chunk: prob, rst_chunk:prob}

        
        # Applying restrictions and then resampling
            # apply the pos, rel, and ns restrictions
                # if any no probabilistic op
        
        # creating filtering kwargs
        filt_kwargs = {}
        if rel!=None: filt_kwargs['rel_filt'] = rel 
        if ns!=None: filt_kwargs['ns_filt'] = ns 

        # filtering conditional distr on rst chunk restrictions
        if len(filt_kwargs)>0:
            li_rsts_rstsdecoded_prob = [ [key, ujson.loads(key), cond_distr[key] ]  for key in cond_distr ]
            li_rsts_rstsdecoded_prob = filter(lambda rst_chunk, rst_chunk_decoded, prob: generation_filter(rst_chunk_decoded, **filt_kwargs ), li_rsts_rstsdecoded_prob)

            # filtered conditional distr
            cond_distr = { rstchunk_encoded:prob for rstchunk_encoded, rstchunk_decoded, prob in  li_rsts_rstsdecoded_prob  }

            # renormalized cond distr
            cond_distr  = self.renormalize_probs(cond_distr)
        
        #reduction of feature space part 1 - removing rst relations to be ingored
        if reduce_rel_space == True:
                                            # Attribution -> Ignore
                                            # Background -> Combine with 
                                            # Cause -> Combine with Explanation, 
                                            # Comparison
                                            # Condition 
                                            # Contrast
                                            # Elaboration -> Ignore
                                            # Enablement
                                            # Evaluation -> Topic-Comment
                                            # Joint 
                                            # Manner-Means
                                            # Summary -> Combine with Topic-Comment
                                            # Textual organisation
                                            # Temporal
                                            # Topic-Change ->Ignore
                                            # n -> Ignore
                                            # same-unit -> Ignore

            ignore_list = ['Attribution','Elaboration','n','same-unit','Topic-Change']
            cond_distr = { k:v for k,v in cond_distr.items() if ujson.loads(k)['rel'] not in ignore_list }
            cond_distr  = self.renormalize_probs(cond_distr)
        
        # Adapting the markov distribution to ensure that cond_rstlength is satisfied
        rst_seq_ex_edus = [ rst_chunk for rst_chunk in rst_seq if rst_chunk['rel'] not in ['start','edu'] ]
        
        # if cond_rstlength != None and len(rst_seq_ex_edus)>cond_rstlength and curr_rstchunk_pos != -1:
        #     break
        if cond_rstlength != None and len(rst_seq_ex_edus)<cond_rstlength and rst_cond_chunk['pos'] != -1:
            # adapts the conditional distribuiton of the curr_rstchunk distr
            child_node_pos = ujson.loads(next(iter(cond_distr)))['pos']
            cond_distr = self.sample_length_enforcer( cond_distr, cond_rstlength, rst_cond_chunk, rst_seq, child_node_pos )
        else:
            cond_distr = cond_distr 
        
        rst_chunks, probs = [ list(x) for x in zip( *list(cond_distr.items()) ) ]
                
        # sampling from conditional distr
        if sampling_method == "greedy":
            max_p = max(probs)
            next_rst_chunk = rst_chunks[ probs.index(max_p) ]
        
        elif sampling_method=="random":
            next_rst_chunk = str( self.rng.choice(rst_chunks, p=probs ) )


        
        #reduction of feature space part 2 - converting rst rel names to reduced space
        if reduce_rel_space == True:
            mapping = {
                'Cause':'Explanation',
                'Summary':'Topic-Comment',
                'Evaluation':'Topic-Comment'}

            decoded_rstchunk = ujson.loads(next_rst_chunk)
            if decoded_rstchunk['rel'] in mapping.keys():
                decoded_rstchunk['rel'] = mapping[ decoded_rstchunk['rel']  ]
                next_rst_chunk = ujson.dumps( decoded_rstchunk )

        return ujson.loads(next_rst_chunk)

    def sample_length_enforcer( self, cond_distr, cond_rstlength, rst_chunk, rst_seq, child_node_pos):

        #markov_distribution1 = copy.deepcopy(markov_distribution)
        #cond_distr

        edu_rst_chunk = copy.deepcopy(self.edu_rst_chunk)
        edu_rst_chunk['pos'] = child_node_pos
        edu_rst_chunk_encoded =ujson.dumps(edu_rst_chunk)

        # if ujson.dumps(rst_chunk) not in markov_distribution1:
        #     return markov_distribution1

        #cond_distr = markov_distribution1[ ujson.dumps(rst_chunk) ]

        # calculate number of possible branches left to sample - This refers to nodes on the child level
            # zero indexed
        child_node_level = math.floor( math.log2(child_node_pos+1) )

        #max_node_in_level_below = 2** (child_node_level+1) -1-1 

        max_node_at_child_level = 2** (child_node_level+1) -1-1 
        min_node_at_child_level =max_node_at_child_level/2
        

        #for child_node_pos at level l, count all non edu nodes on level l with pos<child_node_pos
        left_non_edu_nodes = len( [ rst_chunk for rst_chunk in rst_seq if (rst_chunk['pos']>=min_node_at_child_level) and (rst_chunk['pos']<child_node_pos) and (rst_chunk['rel']!='edu') ] )
        
        #for child_node_pos at level l, count all remaining nodes to fill at level l with pos>child_node_pos
            # This is done by counting parent nodes of items to the right that are not edu and mutliplying by two, then optionally adding one if child_node_pos is odd       
        parent_node = ujson.loads(next(iter(cond_distr)))['pos']
        max_node_parent_level = min_node_at_child_level -1
        parent_nodes_to_the_right = len( [ rst_chunk for rst_chunk in rst_seq if (rst_chunk['pos']>parent_node) and (rst_chunk['pos']<=max_node_parent_level ) and (rst_chunk['rel']!='edu') ] )
        
        right_nodes_remaining = 2*parent_nodes_to_the_right + (child_node_pos%2)

        #downweight prob of edu chunk being sampled next based on
        #  the possible number of other nodes that can still elongate the rst_sample
        count_of_nodes_that_can_elongate = left_non_edu_nodes + right_nodes_remaining
        
        #number of nodes at level l excluding current node
        left_nodes_that_cant_elongate =  (  child_node_pos - min_node_at_child_level) - left_non_edu_nodes
        nodes_at_curr_level = count_of_nodes_that_can_elongate + left_nodes_that_cant_elongate
        ratio_to_reduce_edu_prob = count_of_nodes_that_can_elongate/(nodes_at_curr_level)

        # reducing the relative weight of the probability of the edu rst_chunk
        edu_prob = cond_distr[ edu_rst_chunk_encoded ]
        count_other_rstchunks = len(cond_distr ) -1
        
        if ratio_to_reduce_edu_prob != 1.0:
            for key in cond_distr:
                if key != edu_rst_chunk_encoded:
                    cond_distr[key] = cond_distr[key] + (edu_prob * (1-ratio_to_reduce_edu_prob) ) /count_other_rstchunks
                else:
                    cond_distr[edu_rst_chunk_encoded] = edu_prob * ratio_to_reduce_edu_prob
            
        #markov_distribution1[ ujson.dumps(rst_chunk) ] = cond_distr

        return cond_distr
       

    def generation_filter(self, rst_chunk_decoded, rel_filt=None, ns_filt=None):
        """[Returns True/False depending on whether an RSTchunk satistifies the rst filters]

        Args:
            rst_chunk ([type]): [description]
            pos_filt ([type], optional): [description]. Defaults to None.
            rel_filt ([type], optional): [description]. Defaults to None.
            ns_filt ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        
        if rel_filt!=None and rst_chunk_decoded['rel'] not in rel_filt:
            return False

        if ns_filt!=None and rst_chunk_decoded['ns'] not in ns_filt:
            return False

        return True

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(add_help=True, allow_abbrev=False)
        parser.add_argument( '-mn', '--model_name', default='rst_planner', required=False)
        parser.add_argument('-drc', '--dirs_rst_convo', default=os.path.abspath("./dataset_v2"), required=False)
        parser.add_argument('-mcnp', '--maximum_child_node_pos', default=30)
        parser.add_argument('-df','--data_filters', default="{}" )
        parser.add_argument('-sm','--sampling_params',
                                default= "{ \"sampling_method\":\"random\", \"cond_subreddit\":\"aggregated\", \"cond_rstlength\":7, \"reduce_rel_space\":True }")
        
        mparams = parser.parse_known_args( )[0]

        mparams.data_filters = eval( mparams.data_filters )
        mparams.sampling_params = eval(mparams.sampling_params )
       
        return mparams


def main(mparams):

    sampling_params = mparams.pop('sampling_params',)

    rst_planner = RSTPlanner(**mparams)

    # create markov distributions for each subreddit
    for subreddit in rst_planner.li_subreddit_names:

        rst_planner.create_markovian_distr(subreddit, mparams['data_filters'])
        if subreddit in rst_planner.dict_subreddit_lirstseq:
            del rst_planner.dict_subreddit_lirstseq[subreddit]
    
    # create aggregated version
    rst_planner.create_markovian_distr("aggregated")

    # Generate an rst sequence
    sample_rst_chain = rst_planner.sample_rst_chain( **sampling_params )

    with open( "./RSTMarkov/example_sample.txt", "w" ) as f:
        f.write( ujson.dumps(sample_rst_chain) )

if __name__ == '__main__':
    # add model specific args
    mparams = RSTPlanner.parse_args()
    mparams = vars(mparams)
    main(mparams)