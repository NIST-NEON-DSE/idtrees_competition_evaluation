[![DOI](https://zenodo.org/badge/265101910.svg)](https://zenodo.org/badge/latestdoi/265101910) 

# idtrees_competition_evaluation
Evaluation metrics for 2020 IDTreeS competition using NEON remote sensing data

<h2> IDTreeS Data Science competition for identifying trees from remote sensing </h2>

This repo hosts the code for running the evaluation metrics of the 2020 IDTreeS competition. 
To run it on your local machine be sure to have installed all the requirements for your environment (see installation instructions below); 
you can run the evaluation for both tasks by running the `main(args)` function (`main.py`), for other cases see Examples.

## Installation
### 1) clone repo
```
git clone https://github.com/NIST-NEON-DSE/idtrees_competition_evaluation.git
```
### 2) Initialize environment with requirements
```
conda env create -f environment.yaml
conda activate idtrees
```

*Data for evaluation should be stored as follow:*
- ./RS/RGB folder: contains RGB data that will be used to determine which detections to evaluate for each plot
- ./eval/submission: contains groundtruth and predictions spatial data (multipolygons with coordinates in wkt format)
- ./scores: stores the outputs of the evaluation. Default is storing evaluation metrics as `csv`. Flagging the arguments parameter `save` to 1 in `parameters.py` will also save the plot of groundtruth - detection pairs selected by the hungarian algorithm.



## Examples:
## Run Demo
```python
python evaluation.py
```

## Run task 1
- Task 1 can be exectuted in an (a) IDE or (b) in console.

### a) In python IDE
```python
#outputs will be stored in the scores folder. Evaluation outputs stored in the task1_evaluation.csv file
#save your groundtruth/evaluation set in the submission folder as *_ground.csv (e.g. ./submission/OSBS_ground.csv)
#save your submission file into the submission folder as *_submission.csv  (e.g. ./submission/OSBS_submission.csv)
```
#### Demo
```python
#run the following code:
from parameters import *
from evaluation import *
args = evaluation_parameters(None)
run_segmentation_evaluation(args)
```
#### with arguments
```python
#run the following code:
from parameters import *
from evaluation import *
args = evaluation_parameters(['--datadir',pathtodata,'--outputdir',pathtosave,...])
run_segmentation_evaluation(args)
```

### b) In console
```
python evaluation.py --datadir "folderpath" --outputdir "folderpath" --task "task1" --save boolean (0 or 1)
```

## Run task 2
- Task 2 can be exectuted in an (a) IDE or (b) in console.

### a) In python IDE
```python
#outputs will be stored in the scores folder. Evaluation outputs stored in the task2_evaluation.csv file
#save your groundtruth file into the submission folder as task2_ground.csv  (e.g. ./submission/task2_ground.csv)
#save your submission set in the submission folder as task2_submission.csv (e.g. ./submission/task2_submission.csv)
```
#### Demo
```python
#run the following code:
from parameters import *
from evaluation import *
args = evaluation_parameters(None)
run_classification_evaluation(args)
```
#### with arguments
```python
#run the following code:
from parameters import *
from evaluation import *
args = evaluation_parameters(['--datadir',pathtodata,'--outputdir',pathtosave,...])
run_classification_evaluation(args)
```
### b) In console
```
python evaluation.py --datadir "folderpath" --outputdir "folderpath" --task "task2"
```
