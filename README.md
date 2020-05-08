# idtrees_competition_evaluation
Evaluation metrics for 2020 competition using NEON remote sensing data

<h2> IDTreeS Data Science competition for identifying trees from remote sensing </h2>

This repo hosts the code for running the evaluation metrics of the 2020 IDTreeS competition. 
To run it on your local machine be sure to have installed all the requirements on your environment; 
you can run the evaluation for both tasks by running the `main(args)` function, 
or by calling `run_segmentation_evaluation(args)` for evaluation of task 1  (detectiona nd segmentation)
and `run_classification_evaluation(args)` for task 2 (species classification)

*Data for evaluation should be stored as follow:*
- ./eval/RS/RGB folder: contains RGB data that will be used to determine withi detections to evaluate for each plot
- ./eval/submission: contains groundtruth and predictions spatial data (multipolygons with coordinates in wtk format)

    - save your submission file into the submission folder as *_submission.csv 
      (e.g. ./submission/OSBS_submission.csv)
    - save your groundtruth/evaluation set in the submission folder as *_ground.csv
      (e.g. ./submission/OSBS_submission.csv)
    - make sure you have the RGB of plots in ./RS/RGB/
    
    Run:
        
    evaluation = run_segmentation_evaluation()
