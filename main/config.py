# return hyper parameters

import os

def get_global_parameters():
    dic_param = {
                 'dim_hidden':512, 
                 #'dim_hidden':256,
                 'n_frame_step': 30, ## DO NOT CHANGE
                 'n_epochs': 1000,
                 #'learning_rate': 0.0001,  ## GOOD FOR VGG + RESNET
                 'learning_rate': 0.0001,  ## Test for GoogleNet
                 'batch_size': 16,
                 'resnet':{'dim_image':2048,
                           'train_path': '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/train_resnet50.pkl', 
                           'test_path':'/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/test_resnet50.pkl'},
                 'vgg19':{'dim_image':4096,
                           'train_path': '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/train_vgg19.pkl',  
                           'test_path':'/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/test_vgg19.pkl'},
                 
                 'resnet50_keras_feature_no_sub_mean':{'dim_image':2048,
                           'train_path': '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/train_resnet50_keras_feature_no_sub_mean.pkl', 
                           'test_path':'/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/test_resnet50_keras_feature_no_sub_mean.pkl'},
                 
                 'vgg16_keras_feature_no_sub_mean':{'dim_image':4096,
                           'train_path': '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/train_vgg16_keras_feature_no_sub_mean.pkl', 
                           'test_path':'/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/test_vgg16_keras_feature_no_sub_mean.pkl'},
                 
                 'inception_keras_feature_no_sub_mean':{'dim_image':2048,
                           'train_path': '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/train_inception_keras_feature_no_sub_mean.pkl', 
                           'test_path':'/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/test_inception_keras_feature_no_sub_mean.pkl'},
                 
                 'inception_tensorflow_from_saliency':{'dim_image':2048,
                           'train_path': '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/train_inception_tensorflow_from_saliency.pkl', 
                           'test_path':'/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/test_inception_tensorflow_from_saliency.pkl'},
                 
                 }
    
    return dic_param


    
