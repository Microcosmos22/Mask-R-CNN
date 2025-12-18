# Mask-R-CNN for image-manipulation detection (current state)

This project is part of the Kaggle competition "Recod.ai/LUC" and shows the current state of the project, 
flaws that could be overcome and lays a path for future work once some problems are solved.



### Description of Mask R-CNN

**Mask-R-CNN** is a region based algorithm for detection, classification and segmentation of images.

Its composed of a backbone encoder, a Region Proposal Network (RPN) with Bounding Box (BBox) regression and a mask-head, among others.
Therefore it is also hard to debug which parts are failing, although we managed to narrow the problem down to the Bounding Box regressor inside the RPN.

# Approach:
- Inspired by: https://www.kaggle.com/code/antonoof/eda-r-cnn-model, who however does not present any results, other than one authentic prediction (an maskless image).
- Provides NN architecture, Dataloading pipeline, training loops, but NOT: steps for numerical debugging or step-wise tests for the different NN parts. We could provide some insights.

## General steps:
1. The first step is overfitting the model to a single image: 10017.png (see image below). To simplify the initial task, we freeze part of the network (the mask head).
2. Then we train only the mask segmentation (we freeze the backbone) and try to overfit 5 images.
3. Unfreeze everything, train on the whole dataset.

## Code:
Run

```
pip install -r requirements.txt
python3 edarnn.py
```



for training and
```
python3 encode_submission.py
```
for evaluation (DICE) over a test_dataset

The first image contains two forgery regions masked in yellow (because they have been manipulated within the image). Here we found already some flaws in the pipeline, but most importantly, the model is not able to overfit.


<img src="images/10017.png" width="400">

Trying to track it down, we plotted the Bounding boxes, which after overfit should be the same as the ground truth:

