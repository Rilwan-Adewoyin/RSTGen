# Workflow

# Split Dataset into 12 new files
# Then perform the following operations in parrallel
    # preprocess
    # add DA and RST annotations

# Then reform dataset into one file and shuffle
# Then set 75% as training set and 25% as test




import argparse
import os
import re
import glob
import logging
import subprocess

import multiprocessing as mp
import psutil
"""
code extended from
    https://github.com/PolyAI-LDN/conversational-datasets/blob/master/opensubtitles/create_data.py
"""


def main(argv=None):
    """Run the preprocessing pipeline."""
    args = _parse_args(argv)

    #preprocessing text
    res = preprocess(args)

    #adding DA and RST annotations to each file in the dataset
    res = annotate(args) 

    #multiprocess run
    result = p.run()
    result.wait_until_finish()

def parse_args(argv=None):
    """Parse command-line args."""

    def _positive_int(value):
        """Define a positive integer ArgumentParser type."""
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "Value must be positive, {} was passed.".format(value))
        return value

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir_path",
        default="../../dataset/open_subs", type=str,
        help="The directory for datasets.")

    parser.add_argument(
        "--min_length",
        default=9, type=_positive_int,
        help="The minimum length of an utterance to include.")

    parser.add_argument(
        "--max_length",
        default=200, type=_positive_int,
        help="The maximum length of an utterance to include.")

    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory to write the dataset.")

    parser.add_argument(
        "--dataset_format",
        choices={_TF_FORMAT, _JSON_FORMAT},
        default="TF",
        help="The dataset format to write. 'TF' for serialized tensorflow "
             "examples in TFRecords. 'JSON' for text files with one JSON "
             "object per line.")

    parser.add_argument(
        "--train_split", default=0.9,
        type=float,
        help="The proportion of data to put in the training set.")

    parser.add_argument(
        "--num_shards_test", default=100,
        type=_positive_int,
        help="The number of shards for the test set.")

    parser.add_argument(
        "--num_shards_train", default=1000,
        type=_positive_int,
        help="The number of shards for the train set.")
    
    parser.add_argument(
        "--num_cores", default=psutil.cpu_count(logical = False),
        type=_positive_int,
        help="The number of cores to split preprocessing job over"
    )

    return parser.parse_known_args(argv)

def preprocess(args):

    # Split the file into groups of 100000 lines
    dset_dir = args.dset_dir
    split_dir =f"{dset_dir}/split"
    annotate_dir = f"{dset_dir}/annotated"

    os.mkdir(split_dir)
    os.mkdir(annotated_dir)

    cmd = [ f"split -d 3 -100000 {dset_dir}/en.txt {split_dir}/en-"]
        # Add code to requirements that automatically downloads OpenSubtitles dataset #downloads folder
        # Actually move this to dataset folder then use a shell script, to activate this script and also download the open subtitles data
    sp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    exit_code = sp.wait()

    if exit_code != 0:
        raise OSError(f"Error Code: {exit_code}")

    # On each file
    pattern = ""
    li_fns = _get_fns( f"{split_dir}/*") # [fn for fn in glob.glob(pattern) ]

    #with mp.Pool as p:
    for fn in li_fns:
        with open( fn , "r" ) as _file:
            li_line = _file.readlines()

        idxs_to_remove = []

        for idx, line in enumerate(li_line):
            
            li_line[idx] = _preprocess_line(line)

            skip = _should_skip(line, args.min_length, args.max_length)
            # error that line may be short after preprocessing, may need to change order
            if skip:
                idxs_to_remove.extend(idx)
                continue
            
            
        
        #deleting invalid lines
        for i in sorted(idxs_to_remove, reverse=True):
            del li_line[idx]
        
        with open( fn , "w" ) as _file:
            _file.writelines(li_line)
 
def _get_fns(pattern):
    li_fns = [fn for fn in glob.glob(pattern) ]
    return li_fns

def _should_skip(line, min_length, max_length):
    """Return boolean indicating
        Whether a line should be skipped
    """

    bool_length = len(line) < min_length or len(line) > max_length
    
    # These lines are usually to indicate the speaker
    bool_colon = ( line[-1] == ':' )

    return  bool_length or bool_colon

def _preprocess_line(line):
    line = line.decode("utf-8")

    # Remove the first word if it is followed by colon (speaker names)
    line = re.sub(r'^.*?:', ':')

    # Remove anything between brackets (corresponds to acoustic events).
    line = re.sub(re.compile(".*?\((.*?)\)"), "", line)

    # Strip blanks hyphens and line breaks
    line = line.strip(" -\n")

    return line

def annotate(args):
    """"
        Produces Annotations for the preprocessed OpenSubtitle dataset
    """"
    dset_dir = args.dset_dir
    annotate_dir = f"{dset_dir}/annotated"

    li_fns = _get_fns( f"{annotate_dir}/*" )

    #with mp.Pool as p:
    for fn in li_fns:
        with open(fn,"r") as _file:
            li_line = _file.readlines()

        for idx, line in enumerate(li_line):

            da_act = 
            li_rst = 

def dialogue_act(text):

def rst(text):

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()