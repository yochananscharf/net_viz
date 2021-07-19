from tensorflow.keras.applications.mobilenet import preprocess_input, MobileNet
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from time import sleep

# %matplotlib inline 

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



class Viualize_Net:
    def __init__(self, model=None) -> None:
        if model==None or model=='MobileNet':
             self.model = MobileNet(
                input_shape=(224,224,3), alpha=0.25, depth_multiplier=1, dropout=0.001,
                include_top=False, weights='imagenet', input_tensor=None, pooling=None,
                classes=1000, classifier_activation='softmax'
            )
        elif model== 'VGG16':
            self.model = VGG16(input_shape=(224,224,3),
                include_top=False, weights='imagenet', input_tensor=None, pooling=None,
                classes=1000, classifier_activation='softmax')
        else:
            print('no model')#self.model = model
        self.test_image = None
        self.image_input = None
        self.activation = None

    def load_image(self, path_to_image, preprocess=True, target_size=224):
        try:
            self.test_image = image.load_img(path_to_image,target_size=(target_size, target_size) )
        except:
            print('check image path')
        self.image_input = image.img_to_array(self.test_image)
        self.image_input = np.expand_dims(self.image_input, axis=0)
        self.image_input = preprocess_input(self.image_input)


    def plot_layer_prediction(self, layer_number=-1, path_to_image=None):

        # if path_to_image == None:
        #     if self.test_image == None:
        #         print('send image path')
        #         return
        # else: #
        #     self.load_image(path_to_image)
        #preprocess image
        self.image_input = self.test_image
        print('image_type = ', self.image_input.dtype)
        inp = self.model.input
        out = self.model.layers[layer_number].output

        self.model_intermediate = Model(inp, out)
        
        try:
            self.activation = self.model_intermediate.predict(self.image_input)
        except:
            print('predict failed')
        print('activation shape =',self.activation.shape)

       
        # plt.imshow(activation_img)
        # plt.imshow(self.activation[0,:,:,3])
        # plt.show()
        # streamlit_img = np.array(activation_img*255,dtype=np.uint8)
        for i in range(10):
            activation_channel = i
            activation_img = self.activation[0,:,:,activation_channel]
            st.image(activation_img, clamp=True, caption='activation imgage for filter {}'.format(i), use_column_width='Auto')
        # st.image(activation_img, clamp=True, caption='activation imgage for filter {}'.format(activation_channel))



if __name__ == '__main__':
    st.title('Feature-Map visualizer')
    option = st.sidebar.selectbox(
    
    'Wich Model would you like to visualize?',
    ('VGG16', 'MobileNet'))
    
    st.write('You selected: ', option)
    img_file = None
    while True:

        img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if img_file is not None:
            break    
        sleep(4)
    # st.write(type(img_file))
    image_arr = Image.open(img_file)
    st.image(image_arr)
    image_arr = image_arr.resize((224, 224))
    image_input = image.img_to_array(image_arr)
    image_input = np.expand_dims(image_input, axis=0)
    image_input = preprocess_input(image_input)

   
    viualize_net = Viualize_Net(option)
    viualize_net.test_image = image_input
    viualize_net.plot_layer_prediction(10)
    