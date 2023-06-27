# Energy landscape method incooperating LDA and pairwise MaxEnt model
The scripts for running energy landscape method described in study:“An energy landscape approach reveals the potential key bacteria contributing to the development of inflammatory bowel disease”.

## How to run

### Step1: LDA model and data preprocessing for Pairwise MaxEnt model
- ```execute_LDA.py``` is the main script for this step.  
- run the following command for all three classes:
```shell
python execute_LDA.py <file of all samples> <file 1 of traning input> ...<file N of traning input>
``` 
```shell
python execute_LDA.py all_990samples_genus.csv CD_genus_representive.csv UC_genus_representive.csv nonIBD_genus_representive.csv
```  
- notice that in our study, the data for fitting LDA model (one representive sample for each individual) is different from the data (ten consecutive samples for each individual) to produce Energy Landscape input. For detail please find the paper.

### Step2: Pairwise MaxEnt model and energy landscape
- ```execute_EL.py``` is the main script for this step. 
- ```rownames.txt``` is used to define the name of assemblages, and the rows number inside should be corresponding to the rows number of your binarized data input file.
- run the following command for single class:

```shell
python execute_EL.py --target_path <the path of binarized input file and rownames.txt> \ 
--binarized_file <the binarized input file> \
--save_path <the path to store the results>
```
```shell
python execute_EL.py --target_path ./ \ 
--binarized_file binarized_data_01_CD.csv \
--save_path ./
```  
## Source of original data
The 16S microbiome abundance data are available from study NIDDK U54DE023798: https://ibdmdb.org/tunnel/public/HMP2/16S/1806/products


