# prepare evaluation result to the right format --> feed to cocoeval.py

import os
import sys
import pickle


#net_id = 'vgg19_batchsize16_dimhidden256'
#net_id = 'resnet_batchsize16_dimhidden256'

#net_id = 'restnet_no_mean_batchsize16_dimhidden256_learningrate0.0001'

## REAL EXP
#net_id = 'LSTM_resnet50_keras_feature_no_sub_mean_batchsize16_dimhidden256_learningrate0.0001'
#net_id = 'GRU_resnet50_keras_feature_no_sub_mean_batchsize16_dimhidden256_learningrate0.0001'
#net_id = 'LSTM_vgg16_keras_feature_no_sub_mean_batchsize16_dimhidden256_learningrate0.0001'
#net_id = 'GRU_vgg16_keras_feature_no_sub_mean_batchsize16_dimhidden256_learningrate0.0001'

#net_id = 'LSTM_resnet50_keras_feature_no_sub_mean_batchsize16_dimhidden512_learningrate0.0001'
#net_id = 'GRU_resnet50_keras_feature_no_sub_mean_batchsize16_dimhidden512_learningrate0.0001'
#net_id = 'LSTM_vgg16_keras_feature_no_sub_mean_batchsize16_dimhidden512_learningrate0.0001'
#net_id = 'GRU_vgg16_keras_feature_no_sub_mean_batchsize16_dimhidden512_learningrate0.0001'
#net_id = 'BasicLSTM_resnet50_keras_feature_no_sub_mean_batchsize16_dimhidden512_learningrate0.0001'
#net_id = 'BasicLSTM_vgg16_keras_feature_no_sub_mean_batchsize16_dimhidden512_learningrate0.0001'

#net_id = 'LSTM_inception_keras_feature_no_sub_mean_batchsize16_dimhidden256_learningrate1e-05'

#net_id = 'LSTM_inception_tensorflow_from_saliency_batchsize16_dimhidden512_learningrate0.0001'
#net_id = 'GRU_inception_tensorflow_from_saliency_batchsize16_dimhidden512_learningrate0.0001'
net_id = 'BasicLSTM_inception_tensorflow_from_saliency_batchsize16_dimhidden512_learningrate0.0001'



# in_prediction_file = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/vgg19_batchsize16_dimhidden256/prediction/prediction.txt'
# out_prd_file = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/vgg19_batchsize16_dimhidden256/prediction/dic_prediction.pickle'
# out_grt_file = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/vgg19_batchsize16_dimhidden256/prediction/dic_groundtruth.pickle'

in_prediction_file = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/' + net_id + '/prediction/prediction.txt'
out_prd_file = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/' + net_id + '/prediction/dic_prediction.pickle'
out_grt_file = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/' + net_id + '/prediction/dic_groundtruth.pickle'



# save dic to pickle file
# a = {'hello': 'world'}
# with open(out_prediction_file, 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(out_prediction_file, 'rb') as handle:
#     b = pickle.load(handle)
# print a == b

# read file
in_file = open(in_prediction_file, 'r')
all_lines = in_file.read().split('\n')
#print 'All lines: ', all_lines

gts_dic = {} # groundtruth dic
prd_dic = {} # prediction dic


# gts = {
#     '184321':[
#     {u'image_id': '184321', u'cap_id': 0, u'caption': u'A train traveling down tracks next to lights.',
#      'tokenized': 'a train traveling down tracks next to lights'},
#     {u'image_id': '184321', u'cap_id': 1, u'caption': u'A train coming down the tracks arriving at a station.',
#      'tokenized': 'a train coming down the tracks arriving at a station'}],
#     '81922': [
#     {u'image_id': '81922', u'cap_id': 0, u'caption': u'A large jetliner flying over a traffic filled street.',
#      'tokenized': 'a large jetliner flying over a traffic filled street'},
#     {u'image_id': '81922', u'cap_id': 1, u'caption': u'The plane is flying over top of the cars',
#      'tokenized': 'the plan is flying over top of the cars'},]
#     }


for i in range(0, len(all_lines)-4, 4):
    sp_line = all_lines[i]
    id_line = all_lines[i+1]
    gt_line = all_lines[i+2]
    pd_line = all_lines[i+3]
    
    print 'seperated line: ', sp_line
    print 'id line: ', id_line
    print 'ground truth line: ', gt_line
    print 'prediction line: ', pd_line
    
    # build current grt item
    cur_grt_dic = {}
    cur_grt_dic['image_id'] = id_line
    cur_grt_dic['cap_id'] = 0 # only 1 ground truth caption
    cur_grt_dic['caption'] = gt_line
    # add to main grt dic
    gts_dic[id_line] = [cur_grt_dic]
    #print 'current ground truth dic: ', gts_dic
    
    # build current pre item
    cur_prd_dic = {}
    cur_prd_dic['image_id'] = id_line
    cur_prd_dic['caption'] = pd_line
    #cur_prd_dic['caption'] = gt_line  # TESTING --> SHOULD BE 100%
    
    # add to main pre dic
    prd_dic[id_line] = [cur_prd_dic]
    #print 'current prediction dic: ', prd_dic
      

    #break
    
    
# save dic to pickle file
with open(out_prd_file, 'wb') as handle:
    pickle.dump(prd_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(out_grt_file, 'wb') as handle:
    pickle.dump(gts_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)



    
    





print 'ALL DONE!'
