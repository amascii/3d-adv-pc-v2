import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # reduce tf spam
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import importlib
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import tf_nndistance
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size for attack [default: 5]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--data_dir', default='data', help='data folder path [data]')
parser.add_argument('--dump_dir', default='independent', help='dump folder path [independent]')

parser.add_argument('--add_num', type=int, default=512, help='number of added points [default: 512]')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = os.path.join(FLAGS.log_dir, "model.ckpt")
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DATA_DIR = FLAGS.data_dir


NUM_ADD=FLAGS.add_num

attacked_data_all=joblib.load(os.path.join(DATA_DIR,'attacked_data.z'))

def get_crit_p():
  is_training = False
  with tf.Graph().as_default():
    with tf.device('/gpu:'+str(GPU_INDEX)):
      pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
      is_training_pl = tf.placeholder(tf.bool, shape=())

      pert=tf.get_variable(name='pert',shape=[BATCH_SIZE,NUM_ADD,3],initializer=tf.truncated_normal_initializer(stddev=0.01))
      initial_point_pl=tf.placeholder(shape=[BATCH_SIZE,NUM_ADD,3],dtype=tf.float32)
      point_added=initial_point_pl+pert
      pointclouds_input=tf.concat([pointclouds_pl,point_added],axis=1)
          
      pred, end_points = MODEL.get_model(pointclouds_input, is_training_pl)

      vl=tf.global_variables()
      vl=[x for x in vl if 'pert' not in x.name]
      saver = tf.train.Saver(vl)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ops = {
      'pointclouds_pl': pointclouds_pl,
      'labels_pl': labels_pl,
      'is_training_pl': is_training_pl,
      'pointclouds_input':pointclouds_input,
      'initial_point_pl':initial_point_pl,
      'pert': pert,
      'point_added':point_added,
      'pre_max':end_points['pre_max'],
      'post_max':end_points['post_max'],
      'pred': pred,
    }

    saver.restore(sess,MODEL_PATH)
    #print('model restored!')


    # Critical points:
    attacked_data = attacked_data_all[35][:1]
    crit_p=MODEL.get_critical_points(sess, ops, attacked_data, BATCH_SIZE, NUM_ADD,NUM_POINT)
    return crit_p

    #attack_batch(sess, ops, )

    #return sess, ops  

#    for victim in [5,35,33,22,37,2,4,0,30,8]:#the class index of selected 10 largest classes in ModelNet40 (although there are more classes with 100 samples)
#      attacked_data=attacked_data_all[victim]#attacked_data shape:100*1024*3, but only the first 25 are used later on
#      for j in range(ATT_SIZE//BATCH_SIZE):
#        attack_one_batch(sess,ops,attacked_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
#np.save in DUMP_DIR orig.npy, adv.npy, return adv examples of batch

def get_pred(new_attacked_data, init_points):
  is_training = False
  with tf.Graph().as_default():
    with tf.device('/gpu:'+str(GPU_INDEX)):
      pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, np.shape(new_attacked_data)[1])
      is_training_pl = tf.placeholder(tf.bool, shape=())

      pert=tf.get_variable(name='pert',shape=[BATCH_SIZE,NUM_ADD,3],initializer=tf.truncated_normal_initializer(stddev=0.01))
      initial_point_pl=tf.placeholder(shape=[BATCH_SIZE,NUM_ADD,3],dtype=tf.float32)
      point_added=initial_point_pl+pert
      pointclouds_input=tf.concat([pointclouds_pl,point_added],axis=1)
          
      pred, end_points = MODEL.get_model(pointclouds_input, is_training_pl)

      vl=tf.global_variables()
      vl=[x for x in vl if 'pert' not in x.name]
      saver = tf.train.Saver(vl)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ops = {
      'pointclouds_pl': pointclouds_pl,
      'labels_pl': labels_pl,
      'is_training_pl': is_training_pl,
      'pointclouds_input':pointclouds_input,
      'initial_point_pl':initial_point_pl,
      'pert': pert,
      'point_added':point_added,
      'pre_max':end_points['pre_max'],
      'post_max':end_points['post_max'],
      'pred': pred,
    }

    saver.restore(sess,MODEL_PATH)
    #print('model restored!')

  # Try to predict:
    feed_dict = {
      ops['pointclouds_pl']: new_attacked_data,
      ops['is_training_pl']: is_training,
      ops['initial_point_pl']: init_points
    }
  
    pred_val, input_val = sess.run([ops['pred'], ops['pointclouds_input']], feed_dict=feed_dict)
              
    # Predictions here:
    pred_val = np.argmax(pred_val, 1)
    return pred_val




if __name__=='__main__':
  # get k points
  crit_p = get_crit_p()

  print(f'crit shape: {np.shape(crit_p)}')

  for k in range(250):

    # remove k points
    data = attacked_data_all[35][:1] #1x1024x3
    ids = []
    for i in range(k):
      idx = np.where(data[0] == crit_p[0][i])
      ids.append(idx[0][0])
    new_data = np.delete(data[0], ids, 0) #1x(1024-k)x3

    print(f'ids shape: {np.shape(ids)} ids set: {len(set(ids))} new shape: {np.shape(new_data)}')

    # check prediction
    pred = get_pred([new_data], crit_p)

    if pred[0] == 35:
      print(f' pred: {pred[0]} result: same old')
    else:
      print(f' pred: {pred[0]} result: adv example found!')
