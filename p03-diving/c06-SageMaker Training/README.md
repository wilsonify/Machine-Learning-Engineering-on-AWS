
SageMaker Training and Debugging Solutions
===

This chapter presents the different solutions 
and capabilities available 
when training a machine learning model using Amazon SageMaker. 

Here, we dive a bit deeper into the different options and strategies 
when training and tuning ML models in SageMaker.


A .lst file is a tab-separated file with three columns that contains a list of image files. 

The first column specifies the image index, 
the second column specifies the class label index for the image, 
and the third column specifies the relative path of the image file. 

The image index in the first column must be unique across all images. 

The set of class label indices are numbered successively and the numbering should start with 0. 

For example, 0 for the cat class, 1 for the dog class, and so on for additional classes.

The following is an example of a .lst file:

```
5      1   your_image_directory/train_img_dog1.jpg
1000   0   your_image_directory/train_img_cat1.jpg
22     1   your_image_directory/train_img_dog2.jpg
```