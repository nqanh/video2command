import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras.preprocessing import sequence

import sys
import matplotlib.pyplot as plt

#root_path = '/home/anguyen/workspace/paper_src/2018.icra.aff.source'  # not .source/dataset --> wrong folder
cwd = os.getcwd()
print 'current dir: ', cwd
root_path = os.path.abspath(os.path.join(cwd, os.pardir))  # get parent path
print 'root path: ', root_path  
sys.path.insert(0, root_path)


from dataset.breakfast.data_io import load_data
from main.config import get_global_parameters


rnn_id = 'BasicLSTM'
#rnn_id = 'LSTM' # 'GRU'
#rnn_id = 'GRU'

cnn_name = 'resnet50_keras_feature_no_sub_mean'

from main.main_BasicLSTM_v2c import V2C




def predict(model_path, test_path, dim_hidden, dim_image, n_frame_step, batch_size):
  
    #test_path = '/home/anguyen/workspace/dataset/Breakfast/v2c_dataset/train_test_split/test.pkl'
    print 'test_path: ', test_path
    test_data = load_data(test_path)
    
   
    video_path = test_data.get_value(0, 'video_path')
    video_capt = test_data.get_value(0, 'caption')
   
	ixw_path = os.path.join(root_path, 'output') + '/ixtoword.npy'
    ixtoword = pd.Series(np.load(ixw_path).tolist())
    
  
    model = V2C (dim_image=dim_image,
                n_words=len(ixtoword),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                bias_init_vector=None)
  
    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
      
    # create a session to restore weight
    sess = tf.InteractiveSession()
  
    saver = tf.train.Saver()
  
    #saver.restore(sess, '/home/anguyen/workspace/paper_src/2018.icra.aff.source/output/output_weights/model-900')
    saver.restore(sess, model_path)
    print 'restore model ok!'
    
    out_file = os.path.join(root_path, 'output', net_id, 'prediction') + '/prediction.txt'
    #out_file = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/s2s_batch16_vgg_feature/prediction/prediction.txt'
    fwriter = open(out_file, 'w')
    
    for i in range(len(test_data)):
        
        video_feat_path = test_data.get_value(i, 'video_path')
        video_grth_capt = test_data.get_value(i, 'caption')
        
        print video_feat_path
        video_feat = np.load(video_feat_path)[None,...]
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
  
        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat})
        embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat})
        generated_words = ixtoword[generated_word_index]
  
        punctuation = np.argmax(np.array(generated_words) == '.')
        generated_words = generated_words[:punctuation]
  
        generated_sentence = ' '.join(generated_words)
        
        fwriter.write('------------------------------------------\n')
        fwriter.write(video_feat_path + '\n')
        fwriter.write(video_grth_capt + '\n')    
        fwriter.write(generated_sentence + '\n') 
        
    
if __name__ == "__main__":
    
    dic_param = get_global_parameters()
    dim_hidden= dic_param.get('dim_hidden')
    n_frame_step = dic_param.get('n_frame_step')
    n_epochs = dic_param.get('n_epochs')
    learning_rate = dic_param.get('learning_rate')
    batch_size = dic_param.get('batch_size')
    
    net_dic_param = dic_param.get(cnn_name)
    dim_image = net_dic_param.get('dim_image')
    test_path = net_dic_param.get('test_path')
    

    if rnn_id == 'LSTM':
        net_id = 'LSTM_' + cnn_name + '_batchsize' + str(batch_size) + '_dimhidden' + str(dim_hidden) + '_learningrate' + str(learning_rate)
    else:
        if rnn_id == 'GRU':
            net_id = 'GRU_' + cnn_name + '_batchsize' + str(batch_size) + '_dimhidden' + str(dim_hidden) + '_learningrate' + str(learning_rate)
        else:
            net_id = 'BasicLSTM_' + cnn_name + '_batchsize' + str(batch_size) + '_dimhidden' + str(dim_hidden) + '_learningrate' + str(learning_rate)

    #model_path = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/s2s_batch16_vgg_feature/log_model/model-1300'
    #model_path = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/resnet_batchsize16_dimhidden256/log_model/saved_model-100'
    
	model_path = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/' + net_id + '/log_model/saved_model-150'
    
    predict(model_path, test_path, dim_hidden, dim_image, n_frame_step, batch_size)
    
    print 'ALL DONE!'