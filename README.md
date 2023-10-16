# RSTGen Repository

This is the supporting repository for the paper: [RSTGen: Imbuing Fine-Grained Interpretable Control into Long-Form Text Generators](https://aclanthology.org/2022.naacl-main.133/)

## Important
- Code is currently being refactored.
- To download the dataset, please refer to the Dataset section below.
- For understanding the features of the RSTGen framework as they were implemented, kindly refer to the files in `src/trainers`.
- Code updates for proper functionality are planned over the next month.

## Environment Setup
1. Clone this repository:
   ```
   git clone https://github.com/Rilwan-Adewoyin/RSTGen.git
   ```

2. Create and activate the conda environment:
   ```
   cd ./RSTGen
   conda env create --file conda_environment.yml
   conda activate rstgen
   ```

## Data Download
I provided the post-processed dataset we created below. This downloads RST-Annotated texts across multiple sub-reddits:

1. To download the necessary data files, run the `download_data.py` script:
   ```
   python .data/download_data.py
   ```

   This script will download all the files from the shared [Google Drive link](https://drive.google.com/drive/folders/1seqNux3ycMLl-FMqbDRa3F5LhP7r61vG?usp=sharing). The files, which are compressed with .tar.gz compression, will be decompressed and saved to the local directory `./data/data_files`.

## Training
- (This section is currently empty.)