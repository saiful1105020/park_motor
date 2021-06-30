(i) First, we are focusing on the finger tapping task (Task 2 in Parktest)*.

*** Setup: conda environment: park_motor
*** Notebook Dir: jupyter notebook --notebook-dir="C:\Wasif\PDMotorFeatureExtraction"
*** Data: "C:\Wasif\PD Motor Feature Extraction\Task2_All"

(ii) Convert videos to a fixed frame rate (15 FPS), resize to 256x256, and all videos to .mp4 using ffmpeg, use the notebook code "FingerTapping_Preprocessing.ipynb"
Output is in "./Task2_15_resized_mp4".

(iii) We are using "temporary_detecting_the_hand_present_regions" notebook to run the mediapipe code . This is used to do temporal segmentation. The output is in 
"C:\Wasif\PDMotorFeatureExtraction\TemporalSegmentOutput"

(iv) Background Removal/Spatial segmentation
Used BodyPix from Google. 
pip install tf-bodypix
The codes will be found with the following python command:
tf_bodypix.__file__
We modified the original bodypix package (to make it output a video), and the modified code is available in tf_bodypix/ folder. In case of a fresh installation, we need to replace the original folder with this one.

The input directory is temporal segmentation output "C:\Wasif\PDMotorFeatureExtraction\TemporalSegmentOutput"
The output directory is "C:\Wasif\PDMotorFeatureExtraction\BodyPixOutput"

(v) Running DeepMAG

(vi) Fourier Feature Extraction from Raw Pixels

Code FeatureExtractFromRawPixels.ipynb and Dataset_Preparation.ipynb

(vii) Fourier Feature Extraction from DeepMAG output

Code FeatureExtractFromDeepMAG.ipynb and Dataset_Preparation_DeepMAG.ipynb

(viii) Run simple SVM model on the dataset


Extracted Features
==================
Output is in C:\Wasif\PD Motor Feature Extraction\TASK2_FEATURES_04_21

x_repeat_removed_raw_pixels.npy is of shape (787,65664) 
-- 787 data points
-- first 256x256 are pixel-wise features 
-- last 128 are frequency wise features

y_repeat_removed_raw_pixels.npy is PD labels (1=PD, 0=Non-PD) of shape (787,)

index_repeat_removed.pickle is a list of length 787, containing the file_ids of the videos in the same sequence as the x and ys.

Possible Improvements:
Blurring background -- problem is introduced noise and information loss
Hand detection only -- separate research direction

Results (10-fold CV)
=======
Temporal Only + SMOTE
{'TP': 368, 'TN': 260, 'FP': 232, 'FN': 124, 'precision': 0.6133333333333333, 'recall': 0.7479674796747967, 'f1': 0.6739926739926739, 'accuracy': 0.6382113821138211}
Temporal Only
{'TP': 29, 'TN': 487, 'FP': 5, 'FN': 266, 'precision': 0.8529411764705882, 'recall': 0.09830508474576272, 'f1': 0.1762917933130699, 'accuracy': 0.6556543837357052}
--> Dataset imbalance is a significant problem

Failed Temporal Segmentation
============================
We failed to run temporal segmentation on these videos: (69 Total)
./TemporalSegmentOutput/2018-01-07T16-08-42-890Z47-task2-temp.mp4
./TemporalSegmentOutput/2018-08-31T12-10-03-310Z4-task2-temp.mp4
./TemporalSegmentOutput/2018-09-06T04-39-17-281Z48-task2-temp.mp4
./TemporalSegmentOutput/2018-09-17T16-16-23-122Z91-task2-temp.mp4
./TemporalSegmentOutput/2018-09-17T18-18-32-131Z7-task2-temp.mp4
./TemporalSegmentOutput/2018-09-17T21-51-58-488Z33-task2-temp.mp4
./TemporalSegmentOutput/2018-09-18T00-16-38-599Z60-task2-temp.mp4
./TemporalSegmentOutput/2018-09-18T12-03-32-615Z88-task2-temp.mp4
./TemporalSegmentOutput/2018-09-20T03-27-11-672Z68-task2-temp.mp4
./TemporalSegmentOutput/2018-09-20T03-33-06-378Z23-task2-temp.mp4
./TemporalSegmentOutput/2018-09-22T17-46-22-793Z38-task2-temp.mp4
./TemporalSegmentOutput/2018-09-27T23-18-03-209Z34-task2-temp.mp4
./TemporalSegmentOutput/2018-10-19T03-27-52-634Z31-task2-temp.mp4
./TemporalSegmentOutput/2018-10-22T17-48-10-055Z55-task2-temp.mp4
./TemporalSegmentOutput/2018-10-25T19-25-11-690Z8-task2-temp.mp4
./TemporalSegmentOutput/2018-10-26T18-44-03-289Z88-task2-temp.mp4
./TemporalSegmentOutput/2018-10-28T19-18-36-699Z53-task2-temp.mp4
./TemporalSegmentOutput/2018-11-13T00-10-15-256Z35-task2-temp.mp4
./TemporalSegmentOutput/2018-11-13T05-18-46-207Z86-task2-temp.mp4
./TemporalSegmentOutput/2018-11-27T18-53-52-554Z55-task2-temp.mp4
./TemporalSegmentOutput/2018-12-01T20-16-51-274Z28-task2-temp.mp4
./TemporalSegmentOutput/2018-12-11T15-32-56-915Z9-task2-temp.mp4
./TemporalSegmentOutput/2018-12-18T03-54-15-579Z14-task2-temp.mp4
./TemporalSegmentOutput/2018-12-23T14-00-21-649Z69-task2-temp.mp4
./TemporalSegmentOutput/2019-04-10T15-24-39-451Z48-task2-temp.mp4
./TemporalSegmentOutput/2019-05-14T14-29-43-211Z1-task2-temp.mp4
./TemporalSegmentOutput/2019-09-17T15-17-49-129Z81-task2-temp.mp4
./TemporalSegmentOutput/2019-10-05T14-19-59-179Z31-task2-temp.mp4
./TemporalSegmentOutput/2019-10-12T23-04-05-531Z36-task2-temp.mp4
./TemporalSegmentOutput/2019-10-22T02-51-27-173Z69-task2-temp.mp4
./TemporalSegmentOutput/2019-10-23T10-04-52-316Z81-task2-temp.mp4
./TemporalSegmentOutput/2019-10-24T22-12-57-133Z44-task2-temp.mp4
./TemporalSegmentOutput/2019-10-25T10-46-16-091Z30-task2-temp.mp4
./TemporalSegmentOutput/2019-12-07T12-20-33-746Z58-task2-temp.mp4
./TemporalSegmentOutput/2019-12-16T19-11-59-943Z0-task2-temp.mp4
./TemporalSegmentOutput/2020-01-23T19-00-31-485Z66-task2-temp.mp4
./TemporalSegmentOutput/2020-01-30T02-24-42-376Z4-task2-temp.mp4
./TemporalSegmentOutput/2020-01-30T02-35-44-293Z51-task2-temp.mp4
./TemporalSegmentOutput/2020-01-30T12-51-52-582Z70-task2-temp.mp4
./TemporalSegmentOutput/2020-01-30T14-15-06-668Z66-task2-temp.mp4
./TemporalSegmentOutput/2020-01-30T16-45-47-152Z92-task2-temp.mp4
./TemporalSegmentOutput/2020-01-31T02-55-49-601Z40-task2-temp.mp4
./TemporalSegmentOutput/2020-01-31T16-43-52-201Z52-task2-temp.mp4
./TemporalSegmentOutput/2020-02-28T17-05-42-114Z49-task2-temp.mp4
./TemporalSegmentOutput/2020-02-28T21-41-23-905Z57-task2-temp.mp4
./TemporalSegmentOutput/2020-02-28T22-25-02-338Z55-task2-temp.mp4
./TemporalSegmentOutput/2020-02-28T23-20-18-367Z85-task2-temp.mp4
./TemporalSegmentOutput/2020-02-29T12-53-22-879Z66-task2-temp.mp4
./TemporalSegmentOutput/2020-03-03T18-59-23-991Z86-task2-temp.mp4
./TemporalSegmentOutput/2020-03-06T23-22-58-073Z49-task2-temp.mp4
./TemporalSegmentOutput/2020-03-13T14-50-48-552Z93-task2-temp.mp4
./TemporalSegmentOutput/2020-03-13T18-36-57-625Z89-task2-temp.mp4
./TemporalSegmentOutput/2020-03-16T23-09-46-626Z3-task2-temp.mp4
./TemporalSegmentOutput/2020-04-01T02-37-42-986Z84-task2-temp.mp4
./TemporalSegmentOutput/2020-04-02T02-00-40-778Z74-task2-temp.mp4
./TemporalSegmentOutput/2020-04-05T12-51-33-751Z29-task2-temp.mp4
./TemporalSegmentOutput/2020-04-05T12-56-22-862Z85-task2-temp.mp4
./TemporalSegmentOutput/2020-04-05T17-24-42-531Z85-task2-temp.mp4
./TemporalSegmentOutput/2020-04-12T16-32-53-545Z81-task2-temp.mp4
./TemporalSegmentOutput/2020-04-13T23-22-01-641Z67-task2-temp.mp4
./TemporalSegmentOutput/2020-04-14T20-48-01-423Z27-task2-temp.mp4
./TemporalSegmentOutput/2020-04-17T13-20-21-791Z74-task2-temp.mp4
./TemporalSegmentOutput/2020-04-22T16-24-52-391Z29-task2-temp.mp4
./TemporalSegmentOutput/2020-04-23T12-55-17-855Z91-task2-temp.mp4
./TemporalSegmentOutput/2020-04-23T13-13-07-753Z31-task2-temp.mp4
./TemporalSegmentOutput/2020-05-15T16-43-12-217Z17-task2-temp.mp4
./TemporalSegmentOutput/2020-06-18T17-49-02-568Z97-task2-temp.mp4
./TemporalSegmentOutput/2020-06-25T05-37-40-785Z52-task2-temp.mp4
./TemporalSegmentOutput/2020-08-17T18-20-49-212Z0-task2-temp.mp4

Temporal Segmentation Output is very small --> Error running DeepMAG (30 Total)
===============================================================================
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2017-10-13T10-48-18-968Z90-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2017-12-17T01-45-18-126Z0-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-02-28T22-28-36-931Z7-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-09-04T10-14-25-684Z23-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-09-05T00-05-08-953Z29-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-09-08T05-43-26-253Z82-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-09-08T23-55-38-127Z5-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-09-17T14-58-21-163Z89-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-09-18T19-32-53-731Z33-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-10-17T15-47-01-938Z79-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-10-24T14-51-50-002Z95-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-10-24T20-15-31-699Z31-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-10-26T20-43-15-644Z50-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-11-22T23-37-55-028Z94-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-11-23T00-05-58-028Z51-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-11-23T21-14-24-181Z80-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-11-29T09-30-51-006Z98-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-12-05T22-12-55-426Z75-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-12-17T08-43-42-380Z30-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-12-23T09-18-38-742Z32-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2018-12-24T22-19-59-878Z14-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2019-04-10T04-09-39-496Z81-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2020-01-23T18-53-28-193Z76-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2020-02-14T21-55-22-284Z95-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2020-03-07T00-09-04-926Z61-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2020-03-14T18-55-36-519Z6-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2020-04-05T17-43-14-023Z17-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2020-04-24T15-59-43-744Z53-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/2020-08-22T20-06-41-140Z71-task2-temp.mp4
E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/null-task2-temp.mp4


To-do:

*can be improved: temporal segmentation chops some part where wrist is not found (30). In future, look for a better remedy
Modify the loss function considering ordinal regression
https://www.ethanrosenthal.com/2018/12/06/spacecutter-ordinal-regression/

06/14/2021