#from keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
#from keras_onnx import conv_onnx
import os



print("[INFO] loading Mask model...")
print("[INFO] loading face mask detector model...")
model = load_model('./alexnet/models/CNN_try2_model.h5')
model.load_weights('./alexnet/weights/CNN_try2_weights-improvement-18-0.84.hdf5')
wt_dir = './alexnet/tf_2.0/trained_models'
out_dir = './alexnet/tf_2.0/converted_models'

#for models in os.listdir(wt_dir):
 #   weight_path = os.path.join(wt_dir, models)
#model.load_weights(weight_path)
    #epoch = models.split('-')[2]
model.save('alexnet_keras.h5')
    #conv_onnx(out_dir + '/mask_classifier_vgg16_'+epoch+'_EP.model', out_dir + '/mask_classifier_vgg16_'+epoch+'_EP.onnx')
