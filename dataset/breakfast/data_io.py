# read train/test.txt --> create .pkl file for train and test
import os
import random
import pandas as pd
import numpy as np

random.seed(0)

in_fid_folder = '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/subtitle/l30_srt'

#in_feature_folder = '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/video/vgg19_caffe_feature'
#in_feature_folder = '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/video/resnet50_keras_feature'

#feature_name = 'resnet50_keras_feature_no_sub_mean'
#feature_name = 'vgg16_keras_feature_no_sub_mean'
#feature_name = 'inception_keras_feature_no_sub_mean'
feature_name = 'inception_tensorflow_from_saliency'


#in_feature_folder = '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/video/resnet50_keras_feature_no_sub_mean'
in_feature_folder = '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/video/' + feature_name
    
in_train_txt_file = '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/train.txt'
in_test_text_file = '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/test.txt'

    
def get_caption(video_id):
    pid1, cam, pid2, task, sentence_id = video_id.split('_')
    
    file_name = '/' + pid1 + '_' + cam + '_' + pid2 + '_' + task + '.fid'
    
    subtitle_path = in_fid_folder + file_name
    #print 'subtitle path: ', subtitle_path
    freader = open(subtitle_path)

    lines = freader.read().split('\n')
    #print lines
    
    #print 'sentence id: ', sentence_id
    ind = (int(sentence_id)-1) * 4 + 2  # sentence id starts from 01 but file has 1 more empty group. Each group has 4 lines. 2 is the index of sentence
    cap_cmd = lines[ind]
    #print 'caption: ', cap_cmd
    
    cap_cmd = cap_cmd.rstrip('\r\n')
    cap_cmd = cap_cmd.rstrip('\n\r')
    cap_cmd = cap_cmd.rstrip('\n')
    cap_cmd = cap_cmd.rstrip('\r')
    
    return cap_cmd
    

def save_data(list_data, out_file):
    # a = pd.DataFrame({'A':[0,1,0,1,0],'B':[True, True, False, False, False]})
    # print a
     
    # # slow solution
    # df = pd.DataFrame(columns=('video_id', 'caption'))
    # for i in range(len(list_train)):
    #     df.loc[i] = [list_train[i], 'my sentence']
    #     
    # print df
    
    # faster solution     
    list_video_id = []
    list_caption = []
    for l in list_data:
        print '-------------------------'
        l = l[0:len(l)-4]  # remove ".npy"
        caption_cmd = get_caption(l)
        print 'output caption: ', caption_cmd
        list_caption.append(caption_cmd)
        
        # build full path -- to load later 
        l = in_feature_folder + '/' + l + '.npy'
        list_video_id.append(l)
        print 'input video id: ', l
        
        
        #break
    #print list_video_id_train
        
    df = pd.DataFrame({'video_path':list_video_id, 'caption':list_caption})
    print df
    
    # save to pickle
    df.to_pickle(out_file)
    
    
# Read data    
def load_data(in_file):
    data = pd.read_pickle(in_file)
    return data

        
    

if __name__ == '__main__':
    
    ### TRAIN TEST FORMAT FILE
    ## video_id_1 caption_1
    ## video_id_2 caption_2
    ## video_id_3 caption_3 
        
    # read list train
    list_train = list(open(in_train_txt_file, 'r'))
    list_train = [l.strip() + '.npy' for l in list_train]
    print 'LIST TRAIN[0]: ', list_train[0]  #P45_webcam01_P45_juice_22.npy
     
    list_test = list(open(in_test_text_file, 'r'))
    list_test = [l.strip() + '.npy' for l in list_test]
    print 'LIST TEST[0]: ', list_test[0]
     
    out_folder = '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
         
# #     train_file = out_folder + '/train.pkl'
# #     test_file = out_folder + '/test.pkl'
#   
    train_file = out_folder + '/' + 'train_' + feature_name + '.pkl'
    test_file = out_folder + '/' + 'test_' + feature_name + '.pkl'
     
    save_data(list_train, train_file)
    save_data(list_test, test_file)
#     
    
    print 'all done!'
