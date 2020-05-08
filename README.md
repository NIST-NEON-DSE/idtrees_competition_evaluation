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
- ./scores: stores the outputs of the evaluation. Default is storing evaluation metrics as `csv`. Flagging the arguments parameter `save` to 1 in `parameters.py` will also save the plot of groundtruth - detection pairs selected by the hungarian algorithm.
