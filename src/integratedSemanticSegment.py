import keras_segmentation
from keras_segmentation import predict
import time
import keras
from keras.models import *
from keras.layers import *
from matplotlib import pyplot as plt
from keras_segmentation.models.model_utils import get_segmentation_model
import tensorflow as tf
import platform
import os
import json
from keras_segmentation.train import find_latest_checkpoint

from keras_segmentation.models.resnet50 import get_resnet50_encoder
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1
pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
start = time.time()


def _unet( n_classes , encoder , l1_skip_conn=True,  input_height=512, input_width=704  ):

    img_input , levels = encoder( input_height=input_height ,  input_width=input_width )
    [f1 , f2 , f3 , f4 , f5 ] = levels

    o = f4

    o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([ o ,f3],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([o,f2],axis=MERGE_AXIS ) )
    o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = ( concatenate([o,f1],axis=MERGE_AXIS ) )

    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = ( concatenate([ o ,img_input],axis=MERGE_AXIS )  )

    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D( 64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = ( BatchNormalization())(o)

    o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )



    model = get_segmentation_model(img_input , o )


    return model

def resnet50_unet( n_classes ,  input_height=512, input_width=704 , encoder_level=3):

    model =  _unet( n_classes , get_resnet50_encoder ,  input_height=input_height, input_width=input_width  )
    model.model_name = "resnet50_unet"
    return model

def model_from_checkpoint_path(model, checkpoints_path ):

    assert ( os.path.isfile(checkpoints_path+"_config.json" ) ) , "Checkpoint not found."
    model_config = json.loads(open(  checkpoints_path+"_config.json" , "r" ).read())
    latest_weights = find_latest_checkpoint( checkpoints_path )
    assert ( not latest_weights is None ) , "Checkpoint not found."
    print("loaded weights " , latest_weights )
    model.load_weights(latest_weights)
    return model



def main():

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)



    model = resnet50_unet(3,  input_height=int(512), input_width=int(704)  )


    IMAGE_ORDERING = 'channels_last'
    MERGE_AXIS = -1
    pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    start = time.time()
    batch_size = 2
    steps_multipier = 2048




    _train_images = "preppedData/input_train"
    _train_annotations = "preppedData/output_train"
    _val_images = "preppedData/input_test"
    _val_annotations = "preppedData/output_test"
    _checkpoints_path = "checkpoints/resnet_unet_3"
    print("the info is", _train_images, _train_annotations, _val_images, _val_annotations, _checkpoints_path)

    model.train(
        train_images= _train_images,
        train_annotations = _train_annotations,
        validate = True,
        val_images = _val_images,
        val_annotations = _val_annotations,
        checkpoints_path = _checkpoints_path,
        epochs=10,
        batch_size = batch_size,
        steps_per_epoch = steps_multipier * 2 / batch_size
    )


    end = time.time()
    runtime = end - start
    msg = "The runtime for {func} took {time} seconds to complete"
    print(msg.format(func="Train",
                     time=runtime))

def test():

    checkpoints_path = "checkpoints/resnet_unet_3"
    assert ( os.path.isfile(checkpoints_path+"_config.json" ) ) , "Checkpoint not found."

    latest_weights = find_latest_checkpoint(checkpoints_path)
    print(latest_weights)


if __name__ == "__main__":

    main()
