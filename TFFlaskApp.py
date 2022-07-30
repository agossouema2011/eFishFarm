# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:08:55 2021

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:55:25 2021

@author: Emmanuel Agossou
This app help to classify a fruit image using Fruits 360 Dataset and ANN
"""
import numpy as np
import   tensorflow,pickle
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import flask,werkzeug
from matplotlib.pyplot import imread
from werkzeug.utils import secure_filename
import os, scipy.misc, tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from PIL import Image
# Creating a new Flask web application. It accepts the package name
app = flask.Flask("FlaskApp")


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def unpickle_patch(file): 
    patch_bin_file =open(file, 'rb') #Reading the binary file.
    patch_dict = pickle.load(patch_bin_file,encoding="bytes")#Loading the details of the binary file into a dictionary.
    return patch_dict #Returning the dictionary


def FishDiseasePredict(img):   
    global sess 
    global graph
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
        
        if result[0][0]>result[0][1]:
            pourcentage=result[0][0]*100
            output="Epizootic ulcerative syndrome (EUS) "+str(pourcentage)+"%\n-Remove the diseased fish from the pond and impove the water quality."
        elif result[0][1]>result[0][0]:
            pourcentage=result[0][1]*100
            output="Ichthyophthirius multifiliis (Ich) "+str(pourcentage)+"%\n-Remove the diseased fish from the pond and impove the water quality."
        else:
            output="La Maladie n'est Pas reconnue"
      
        
        return output

   

def CNN_predict():
    """
Reads the uploaded the diseased fish image file and predicts the disease
using the saved pretrained CNN model.
:return: Either an error if the image disease is not for the
 dataset or redirects the browser to a new page
to show the prediction result if no error occurred.
"""
    global sess 
    global graph
    
    # Setting the previously created 'secure_filename' to global.
    # This is because to be able to invoke a global variable created in another 
    # function, it must be defined global in the caller function.
    global secure_filename
    
    img = Image.open(os.path.join(app.root_path,secure_filename))
    predicted_class = FishDiseasePredict(img)
    return flask.render_template(template_name_or_list="prediction_result.html", predicted_class=predicted_class) 
    '''
    # Checking whether the image dimensions match the Dataset specifications.
    # Dataset images are RGB (i.e. they have 3 dimensions). 
    # Its number of dimensions was not equal to 3, then a message will be returned.
    #img.show()
    if(img.ndim) == 3: 
        # Checking if the number of rows and columns of the read image matched Dataset (32 rows and 32 columns).
       
        if img.shape[0] == img.shape[1] and img.shape[0] == 32: 
            
            # Checking whether the last dimension of the image has just 3 channels (Red, Green, and Blue).
            if img.shape[-1] == 3:
                
                # Passing all preceding conditions, the image is proved to be good for prediction.
                # This is why it is passed to the predictor.
                predicted_class = FishDiseasePredict.main(sess, graph,img) 
                
                # After predicting the diseased class label of the input image, 
                # the prediction disease name is rendered on an HTML page.
                # The HTML page is fetched from the /templates directory. The HTML page accepts 
                # an input which is the predicted class.
                
                return flask.render_template(template_name_or_list="prediction_result.html", predicted_class=predicted_class) 
            else:
                # If the image dimensions do not match the Dataset specifications,
                # then an HTML page is rendered to show the problem.
                return flask.render_template(template_name_or_list="error.html", img_shape=img.shape) 
        else: 
            # If the image dimensions do not match the DataSet specifications,
            # then an HTML page is rendered to show the problem.
            return flask.render_template(template_name_or_list="error.html", img_shape=img.shape) 
    return "An error different from a wrong image dimension has occurred." #Returned if there is a different error other than wrong image dimensions.
    '''
   
# Creating a route between the URL (http://localhost:7777/predict) to a viewer function
# that is called after navigating to such URL.
# Endpoint 'predict' is used to make the route reusable without hard-coding it later.

app.add_url_rule(rule="/predict",endpoint="predict", view_func=CNN_predict)




def upload_image():
    """
Viewer function that is called in response to
getting to the 'http://localhost:7777/upload' URL.
It uploads the selected image to the server.
:return: redirects the application to a new page for
predicting the class of the image.
"""
#Global variable to hold the name of the image file
#for reuse later in prediction by the 'CNN_predict' viewer functions.
    global secure_filename 
    if flask.request.method =="POST": #Checking of the HTTP method initiating the request is POST.
        img_file = flask.request.files["image_file"]
        #secure_filename =werkzeug.secure_filename(img_file.filename) #Getting a secure file name. It is a good practice to use it.
        secure_filename =img_file.filename
        img_file.save(secure_filename) #Saving the image in the specified path.
        print("Image uploaded successfully.") 
        # After uploading the image file successfully, next is to predict the 
        # class label of it. The application will fetch the URL that is tied to the HTML page
        # responsible for prediction and redirects the browser to it.
        # The URL is fetched using the endpoint 'predict'.
        return flask.redirect(flask.url_for(endpoint="predict"))
    return "Image upload failed."

""" Creating a route between the URL
(http://localhost:7777/upload) to a viewer function
that is called after navigating to such URL.
# Endpoint 'upload' is used to make the route
reusable without hard-coding it later. The set of HTTP
method the viewer function is to respond to is added
using the ‘methods’ argument. In this case, the
function will just respond to requests of the methods
of type POST."""

app.add_url_rule(rule="/upload", endpoint="upload",view_func=upload_image, methods=["POST"])




def redirect_upload():  
    """
A viewer function that redirects the Web application
from the root to an HTML page for uploading an image
to get classified.
The HTML page is located under the /templates
directory of the application.
:return: HTML page used for uploading an image. It
is 'upload_image.html' in this example.
"""
    return flask.render_template(template_name_or_list="upload_image.html")
 
"""
Creating a route between the homepage URL
(http://localhost:7777) to a viewer function that is
called after getting to such a URL.
# Endpoint 'homepage' is used to make the route
reusable without hard-coding it later.
"""   
app.add_url_rule(rule="/",endpoint="homepage", view_func=redirect_upload)




def prepare_TF_session(saved_model_path): 
    global sess 
    global graph
    
    sess = tf.Session()

    saver =tf.train.import_meta_graph(saved_model_path+'the_model.meta') 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #Initializing the variables.
        graph = tf.get_default_graph() 
        return graph

# To activate the web server to receive requests,the application must run.
# A good practice is to check whether the file is called from an external Python file or not.
# If not, then it will run.

if __name__ == "__main__":
    # In this example, the app will run based on the following properties: 
    # host: localhost
    # port: 7777
    # debug: flag set to True to return debugging information.
    #Restoring the previously saved trained model.
    
    prepare_TF_session(saved_model_path='Output/') # Directory where the model train model graph is
    app.run(host="127.0.0.5", port=7777, debug=True)
