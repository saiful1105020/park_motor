from oct2py import Oct2Py
oc = Oct2Py()

video = "C:/Wasif/PDMotorFeatureExtraction/BodyPixOutput/2017-10-12T20-07-10-147Z53-task2-temp.mp4"
code_path = "C:/Wasif/PDMotorFeatureExtraction/DeepMAG"

#Add path to MATLAB

#Run nnCropping_br_Hand on video
#nnCropping_br_Hand( video )

oc.runCode(video)