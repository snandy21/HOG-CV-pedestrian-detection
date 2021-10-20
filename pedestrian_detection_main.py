"""
Pedestrian Detection with OpenCV
Histogram of Oriented Gradients (HOG) with OpenCV
SVM-Linear kernel Used
Author : Suprojit Nandy
"""

import cv2
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
from dataPath import DATA_PATH

start_time = time.time() # Program Training Start Point

# Indicate the prediction labels :
# prediction labels for patch
prediction_label = {1:'Pedestrian Detected', -1:'No Pedestrian Detected'}
path_pedestrian_pos_images_train = DATA_PATH + "INRIA_dataset/INRIAPerson/train_64x128_H96/posPatches"
path_pedestrian_neg_images_train = DATA_PATH + "INRIA_dataset/INRIAPerson/train_64x128_H96/negPatches"
path_pedestrian_pos_images_test  = DATA_PATH + "INRIA_dataset/INRIAPerson/test_64x128_H96/posPatches"
path_pedestrian_neg_images_test  = DATA_PATH + "INRIA_dataset/INRIAPerson/test_64x128_H96/negPatches"

# The Data Handling function here :
# Function to parse all different types of images for training here :
def TRAIN_DATA_PARSING(path,file_extension):
    # Append all image paths in one list:
    data_img_path_full = []
    for fil in os.listdir(path):
        img_full_path = os.path.join(path, fil)
        if os.path.splitext(img_full_path)[1] in file_extension:
            data_img_path_full.append(img_full_path)
        #print(os.path.splitext(img_full_path))
    #print(img_path)
    return data_img_path_full
    #sys.exit()


def TRAIN_TEST_DATA_PREP(data_dir_path, class_label):
    img_data      = []
    data_labels   = []
    img_extensions = ['.png','.jpg','.jpeg'] #different training/test image that can be parsed
    data_set_paths = TRAIN_DATA_PARSING(path=data_dir_path, file_extension=img_extensions)
    for counter,img_path in enumerate (data_set_paths):
        img = cv2.imread(img_path)
        img_data.append(img)
        data_labels.append(class_label)
    # print(img_train)
    # print("\n")
    # print(len(data_set_paths)) # number of training images
    # print(np.array(img_train[0]).shape)
    return img_data, data_labels

def SET_SVM_INIT(C_param, gamma_param):
    model_pedestrian_SVM = cv2.ml.SVM_create() # Create the model here
    model_pedestrian_SVM.setC(C_param)
    model_pedestrian_SVM.setGamma(gamma_param)
    model_pedestrian_SVM.setKernel(cv2.ml.SVM_LINEAR)# Use SVM Radial Basis Function and validate
    #model_pedestrian_SVM.setKernel(cv2.ml.SVM_RBF) # Use the SVM Radial
    model_pedestrian_SVM.setType(cv2.ml.SVM_C_SVC)
    #model_pedestrian_SVM.setTermCriteria((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-3))

    return model_pedestrian_SVM

def SVM_TRAIN(model_pedestrian_SVM, data_input, label):
    model_pedestrian_SVM.train(data_input, cv2.ml.ROW_SAMPLE, label)
    return  model_pedestrian_SVM

def SVM_PREDICT(model_pedestrian_SVM, test_data_in, test_labels):
    test_counters = 0
    predictions = model_pedestrian_SVM.predict(test_data_in)
    print("Predictions on the test dataset: \n")
    print(predictions) #The Second index hosts the predictions
    predictions_arr = np.array(predictions[1]).flatten()
    # print("test dimensions = {}\n".format(len(test_labels)))
    # print("prediction dimesions = {}\n".format(len(predictions_arr)))
    # sys.exit()
    for i in range (len(predictions_arr)):
        if int(predictions_arr[i]) == int(test_labels[i]):
            test_counters += 1
    accuracy = (test_counters/len(predictions_arr))*100
    print("Test Set Accuracy achieved = {} \n".format(accuracy))
    return accuracy
    #sys.exit()
# Function for HOG computation :

def HOG_Image(hog, data_in):
    hog_data = []
    # traverse through the data :
    for cntrs,img_in in enumerate (data_in):
        features_HOG = hog.compute(img_in)
        #print("HOG computed for image num = {}\n".format(cntrs))
        hog_data.append(features_HOG)

    return hog_data

def HOG_SVM_DATA_PARSE(hog_data_in): # Convert/reshape HOG features for SVM detection
    print(hog_data_in)
    hog_svm_data_mod = np.array(hog_data_in).reshape(-1,np.array(hog_data_in).shape[1]) # Just keep the num features
    print(hog_svm_data_mod.shape)
    return hog_svm_data_mod

# HOG Features for pedestrian detection :
win_size                = (64,128) # Image Sample size similar with that of Dalal and Triggs Paper
block_size              = (16,16)  # Described in Dalal and Triggs Paper
block_stride            = (8,8)
cell_size               = (8,8)
nbins                   = 9
# derivAperture           = 1
# winSigma                = -1 # Documentation
# histogramNormType       = 0
# L2HysThreshold          = 2.0000000000000001e-01
# gammaCorrection         = 0
nlevels                 = 64

# Evaluate the HOG values for Training and Test Images :
train_pedestrians_hog            = []
train_pedestrians_labels         = []
train_pedestrians_hog_neg        = []
train_pedestrians_hog_neg_labels = []
test_pedestrians_hog             = []
test_pedestrians_hog_labels      = []
test_pedestrians_hog_neg         = []
test_pedestrians_hog_neg_labels  = []
test_dataset = []
test_labels  = []
dataset_dir = os.path.join(DATA_PATH,"INRIA_dataset/INRIAPerson/")
training_path = os.path.join(dataset_dir, "train_64x128_H96")
test_path = os.path.join(dataset_dir, "test_64x128_H96")

# TRAIN MODEL HERE :
pos_training_path = os.path.join(training_path,"posPatches")
neg_training_path = os.path.join(training_path,"negPatches")
print("Prepare the Positive Training Dataset\n")
for i in tqdm(range(100)):
    pedestrian_data_pos_train, pos_train_labels = TRAIN_TEST_DATA_PREP(data_dir_path=pos_training_path,class_label=1)
    time.sleep(0.03)
print("Positive Training Dataset HOG Complete\n")
print("Time taken for positive training images = {}\n".format(time.time()-start_time))
start_neg_HOG_time = time.time()
print("Prepare the negative training dataset HOG features \n")
for k in tqdm(range(100)):
    pedestrian_data_neg_train, neg_train_labels = TRAIN_TEST_DATA_PREP(data_dir_path=neg_training_path,class_label=-1)
    time.sleep(0.03)
print("Negative Training Dataset HOG features complete \n")
print("Time taken for negative training images = {}\n".format(time.time()-start_neg_HOG_time))
train_data_fin = np.concatenate((np.array(pedestrian_data_pos_train), np.array(pedestrian_data_neg_train)),axis=0)# The concatenate the training dataset
train_pedestrians_labels = np.concatenate((np.array(pos_train_labels),np.array(neg_train_labels)),axis=0) # Train labels
print("Training Data Prep Complete \n")
## Create the test data set :
pos_test_path = os.path.join(test_path,"posPatches")
neg_test_path = os.path.join(test_path,"negPatches")
test_pedestrians_hog, test_pedestrians_hog_labels = TRAIN_TEST_DATA_PREP(data_dir_path=pos_test_path, class_label=1)
test_pedestrians_hog_neg, test_pedestrians_hog_neg_labels = TRAIN_TEST_DATA_PREP(data_dir_path=neg_test_path, class_label=-1)
# Prepare the Test dataset:
test_dataset = np.concatenate((np.array(test_pedestrians_hog), np.array(test_pedestrians_hog_neg)), axis=0)
test_labels  = np.concatenate((np.array(test_pedestrians_hog_labels),np.array(test_pedestrians_hog_neg_labels)),axis=0)


# train_pedestrians_hog, train_pedestrians_labels = TRAIN_TEST_DATA_PREP(data_dir_path=path_pedestrian_pos_images_train, class_label=1)
# train_pedestrians_hog_neg, train_pedestrians_hog_neg_labels = TRAIN_TEST_DATA_PREP(data_dir_path=path_pedestrian_neg_images_train, class_label=-1)
# test_pedestrians_hog, test_pedestrians_hog_labels = TRAIN_TEST_DATA_PREP(data_dir_path=path_pedestrian_pos_images_test,class_label=1)
# test_pedestrians_hog_neg, test_pedestrians_hog_neg_labels = TRAIN_TEST_DATA_PREP(data_dir_path=path_pedestrian_neg_images_test,class_label=-1)
#
# training_data_fin = np.concatenate((np.array(train_pedestrians_hog),np.array(train_pedestrians_hog_neg)),axis=0)
# test_data_fin     = np.concatenate((np.array(test_pedestrians_hog), np.array(test_pedestrians_hog_neg)),axis=0)

# Create the HOG descriptor object :
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, nlevels)
# Generate the HOG features for the training images :
print("Training Data-Set HOG features compute starts\n")
for i2 in tqdm(range(100)):
    hog_training_data = HOG_Image(hog=hog,data_in=train_data_fin)
    time.sleep(0.03)
print("HOG features computation for Training Dataset Done\n")
print("Time taken for computation of HOG features for entire dataset={}\n".format(time.time()-start_time))
#hog_test_data     = HOG_Image(hog=hog,data_in=test_data_fin)
print("HOG features for Test Data Set: \n")
for k2 in tqdm(range(100)):
    hog_test_data     = HOG_Image(hog=hog,data_in=test_dataset)
    time.sleep(0.03)
print('HOG features computation done for Negative Dataset \n')
hog_svm_data_in = HOG_SVM_DATA_PARSE(hog_data_in=hog_training_data) # In structure to feed into SVM Module
# print(hog_svm_data_in)
# print(np.array(hog_svm_data_in).shape)
hog_svm_data_in_test = HOG_SVM_DATA_PARSE(hog_data_in=hog_test_data) # Similar to fit into SVM module
# set up model SVM here :
print("**********************\n")
print("**********************\n")
print("TRAINING the model now \n")
for j in tqdm(range(100)):
    model_master = SET_SVM_INIT(C_param=0.1,gamma_param=10) # Set up model here for Linear Kernel
    model_master = SVM_TRAIN(model_pedestrian_SVM=model_master, data_input=hog_svm_data_in,label=train_pedestrians_labels)
    time.sleep(0.03)
# Save the pedestrian models here ::
model_master.save("pedestrian_detection_HOG_SVM.yml") # save the pedestrian detector model (YML) format
print("Model saved in the same host directory \n")
print("TRAINING Complete \n")
print("**********************\n")
print("**********************\n")
print("Now Validate on Test Dataset: \n")
# Check and validate the predictions here :
predictions_accuracy = SVM_PREDICT(model_pedestrian_SVM=model_master,test_data_in=hog_svm_data_in_test,test_labels=test_labels)
print("The final accuracy on test data set = {}\n".format(predictions_accuracy))
# Referto the saved model :
model_SVM_HOG_test = cv2.ml.SVM_load("pedestrian_detection_HOG_SVM.yml")
# get the supprot vectors now :
support_vector_points = model_SVM_HOG_test.getSupportVectors()
rho_param, alpha_param, svidx_param = model_SVM_HOG_test.getDecisionFunction(0)# If the problem solved is regression, 1-class/2-class classification, index = 0
print("Rho param and Rho Shape \n")
print(rho_param, "\n", np.array(rho_param).shape)
print("alpha parameter \n")
print(alpha_param, "\n",np.array(alpha_param).shape)
print("SVIDX param \n")
print(svidx_param, "\n", np.array(svidx_param).shape)
print("Support Vector Points \n")
print(support_vector_points)
print(np.array(support_vector_points).shape)
print(-support_vector_points[:])
# modify Supprt vector dimensions to be fed into HOG features descriptor
svm_HOG_detector = np.zeros(shape=np.array(support_vector_points).shape[1] + 1, dtype=np.array(support_vector_points).dtype) # The feature description length = 3780, +1 for the rho parameter.
#print(np.array((support_vector_points).flatten()).shape[0])
#sys.exit()
#svm_HOG_detector = np.zeros(shape=np.array((support_vector_points).flatten()).shape[0] + 1, dtype=np.array(support_vector_points).dtype)
#svm_HOG_detector = np.zeros(shape=np.array((support_vector_points)))
svm_HOG_detector[:-1] = - support_vector_points[:] # Neg coeff of all numbers
#print(np.array(svm_HOG_detector).shape)
#svm_HOG_detector[:-1] = - np.array(support_vector_points).flatten()[:]/+9*

svm_HOG_detector[-1] = rho_param # Append to the end after multiplying all by -1 (support vector points)
# SVM - HOG Interface in hog object :
print(svm_HOG_detector)
#sys.exit()
hog.setSVMDetector(svm_HOG_detector)
"""
TEST - PEDESTRIAN DETECTION CASE 1 and CASE 2 
"""
input_test = cv2.imread("pedestrians/race.jpg",cv2.IMREAD_COLOR)
# Scale Down Image :
display_img_dimension = 800
scale = (display_img_dimension/np.array(input_test).shape[0])
input_test = cv2.resize(input_test, dsize=None, fx=scale, fy=scale)
print(np.array(input_test).shape)
bouning_box_list,_ = hog.detectMultiScale(img=input_test,hitThreshold=1.0,finalThreshold=2.0, winStride=(4,4), padding=(32,32),scale=1.01)
print(bouning_box_list)
print("The bounding box array dimensions = {}\n".format(np.array(bouning_box_list).shape))
# Display the bounding boxes here :
for pedestrians_cntrs, detection_box in enumerate (bouning_box_list):
    x1, y1, w, h = detection_box
    x2 = x1 + w; y2 = y1 + h
    cv2.rectangle(img=input_test,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,100),thickness=3,lineType=cv2.LINE_AA)

# Display the image with the:
print("NUmber of Pedestrians detected = {}\n".format(pedestrians_cntrs))
cv2.imshow(winname="TEST IMAGE HERE",mat=input_test) # Display the image
cv2.waitKey(0)
cv2.destroyAllWindows()
# returms the bounding box addresses:
#for pedestrian_cntrs, boxes in bouning_box:

"""
TEST PEDESTRIAN DETECTION CASE - 2
"""

img_test_2  = cv2.imread("pedestrians/3.jpg", cv2.IMREAD_COLOR)
# SCALE DOWN IMAGE DIMENSION
scale_dim   = 800
scale_coeff =int(scale_dim/np.array(img_test_2).shape[0])
scale_test_img = cv2.resize(img_test_2,dsize=None, fy= scale_coeff, fx=scale_coeff)
bounding_box_list2,_ = hog.detectMultiScale(img=scale_test_img,hitThreshold=1.0,winStride=(4,4),padding=(32,32),scale=1.2,finalThreshold=2.0)

print(bounding_box_list2)
print(np.array(bounding_box_list2).shape)
for cntrs_test2,detection_box_2 in enumerate (bounding_box_list2):
    x1,y1,w,h = detection_box_2
    x2 = x1 + w; y2= y1 + h
    print((x1,y1),(x2,y2))
    #cv2.rectangle(img=scale_test_img,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,100),thickness=3,lineType=cv2.LINE_AA)
    cv2.rectangle(img=scale_test_img,pt1=(x1,y1),pt2=(x2,y2),color=(0,0,255),thickness=3,lineType=cv2.LINE_AA)

cv2.imshow("Pedestrian Test Image 2",scale_test_img)
cv2.waitKey(0)
print("Number of pedestrians in Test Image = {}\n".format(cntrs_test2))

"""
TEST CASE - 3 - PEDESTRIAN DETECTION 
"""
img_test_3       = cv2.imread("pedestrians/5.jpg",cv2.IMREAD_COLOR)
# Rescale the Image for display
scale_coeff_3    = 800 / np.array(img_test_3).shape[0]
scale_test_img_3 = cv2.resize(src=img_test_3,dsize=None,fx= scale_coeff_3, fy=scale_coeff_3)
# Create HOG Multiscale object

bounding_box_list3,_ = hog.detectMultiScale(img=scale_test_img_3,hitThreshold=1.0,winStride=(4,4),padding=(32,32),scale=1.05,finalThreshold=2.0)

# Establish the bounding box arrangements :
for cntrs_test3, detection_box_3 in enumerate (bounding_box_list3):
    x1, y1, w, h = detection_box_3
    x2 = x1 + w; y2 = y1 + h
    cv2.rectangle(img=scale_test_img_3,pt1=(x1,y1),pt2=(x2,y2),color=(120,180,255),thickness=3,lineType=cv2.LINE_8)

print("The Number of Pedestrians = {}\n".format(cntrs_test3))
cv2.imshow("TEST IMAGE 3",scale_test_img_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
sys.exit()
#TRAIN_TEST_DATA_PREP(data_dir_path=path_pedestrian_pos_images, class_label= 1)
#TRAIN_TEST_DATA_PREP(data_dir_path=path_pedestrian_neg_images, class_label= -1)
#
# Prepare The Training Data arrays/list structures :

#
# # HOG FEATURES FOR PEDESTRIAN DETECTION :
# win_size     = (64,128)
# block_size   = (16,16)
# block_stride = (8,8)
# cell_size    = (8,8)
# n_pyramids   = 8
# n_bins       = 9 # dalal and triggs paper
#
# #print(path_pedestrian_pos_images)
# """
# Data Loader Function Here For Training Images
# """
#
# def train_test_data(path_in):
#
#     train_data = []
