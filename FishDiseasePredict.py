# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:55:03 2021

@author: Emmanuel Agossou
"""
import numpy as np
import   tensorflow,pickle
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def unpickle_patch(file): 
    patch_bin_file =open(file, 'rb') #Reading the binary file.
    patch_dict = pickle.load(patch_bin_file,encoding="bytes")#Loading the details of the binary file into a dictionary.
    return patch_dict #Returning the dictionary

 

def main(sess, img):   
    '''
	 Use an image for prediction
    '''
    img_size=32
    num_channels=3
    images = []
    #image =cv2.imread("train_for_2_Classes32x32/54.jpeg") 
    # Resizing the image to our desired size and
    # preprocessing will be done exactly as done during training
    #image = cv2.resize(img, (img_size, img_size), cv2.INTERFF_LINEAR)
   
    image = img.resize((img_size, img_size))
    #images.append(image)
    #image.show()
    images = np.array(image, dtype=np.uint8)
    images = np.array(images,dtype=np.float32)
    images = np.multiply(images, 1.0/255.0) 
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, img_size,img_size,num_channels)
 
    saver = tf.train.import_meta_graph('Output/the_model.meta')
    graph = tf.get_default_graph()
 
    y_pred = graph.get_tensor_by_name("y_pred:0")
 
    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("xx:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, 2)) 
 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    
   
    init = tf.global_variables_initializer()

    #def train(num_iteration):
    with tf.Session() as sess:
        sess.run(init)
        output=""
        pourcentage=0
        result=sess.run(y_pred, feed_dict=feed_dict_testing)
        print("Prediction Result:",result)
        ''''
        if result[0][0]>result[0][1]:
            pourcentage=result[0][0]*100
            output="Epizootic ulcerative syndrome (EUS) "+str(pourcentage)+"%"
        elif result[0][1]>result[0][0]:
            pourcentage=result[0][1]*100
            output="Ichthyophthirius multifiliis (Ich) "+str(pourcentage)+"%"
        else:
            output="La Maladie n'est Pas reconnue"
        '''
        
        return result

   
