# idtrees_competition_evaluation
Evaluation metrics for 2020 competition using NEON remote sensing data

<h2> IDTreeS Data Science competition for identifying trees from remote sensing </h2>

This repo hosts the code for running the evaluation metrics of the 2020 IDTreeS competition. 
To run it on your local machine be sure to have installed all the requirements on your environment; 
you can run the evaluation for both tasks by running the `main(args)` function, 
or by calling `run_segmentation_evaluation(args)` for evaluation of task 1  (detection and segmentation)
and `run_classification_evaluation(args)` for task 2 (species classification)

## Installation
### 1) initialize a repo in desired folder
```
git init --bare
 ```
### 2) clone repo to folder
```
git clone https://github.com/NIST-NEON-DSE/idtrees_competition_evaluation.git
```
### 3) initialize environment with requirements
```
conda create --name idtrees --file requirements.txt
conda activate idtrees #possible to use source activate idtrees
```
*Data for evaluation should be stored as follow:*
- ./eval/RS/RGB folder: contains RGB data that will be used to determine withi detections to evaluate for each plot
- ./eval/submission: contains groundtruth and predictions spatial data (multipolygons with coordinates in wtk format)
- ./scores: stores the outputs of the evaluation. Default is storing evaluation metrics as `csv`. Flagging the arguments parameter `save` to 1 in `parameters.py` will also save the plot of groundtruth - detection pairs selected by the hungarian algorithm.



Examples: 


# run task 1
```python
#outputs will be stored in the scores folder. Evaluation outputs stored in the task1_evaluation.csv file
#save your groundtruth/evaluation set in the submission folder as *_ground.csv (e.g. ./submission/OSBS_ground.csv)
#save your submission file into the submission folder as *_submission.csv  (e.g. ./submission/OSBS_submission.csv)

#run the following code:
args = evaluation_parameters(None)
run_segmentation_evaluation(args)
```
# run task 2
```python
#outputs will be stored in the scores folder. Evaluation outputs stored in the task2_evaluation.csv file
#save your groundtruth file into the submission folder as task2_ground.csv  (e.g. ./submission/task2_ground.csv)
#save your submission set in the submission folder as task2_submission.csv (e.g. ./submission/task2_submission.csv)
#run the following code:
args = evaluation_parameters(None)
run_classification_evaluation(args)
```
