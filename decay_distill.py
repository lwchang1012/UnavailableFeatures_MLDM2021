import os, time, argparse
from functools import partial

import numpy as np
import tensorflow as tf

from utils import get_data_processor
from config import NUM_CLASS, MISS_CONFIG
from layers import DropoutDense


def parse_args():
    parser = argparse.ArgumentParser('')
    # data settings
    parser.add_argument('--dataset', default='letter')
    parser.add_argument('--miss-type', default='random')
    parser.add_argument('--miss-ratio', type=float, default=0.2)
    parser.add_argument('--miss-config', default='two')
    # experiment settings
    parser.add_argument('--num-exp', type=int, default=10)
    # training parameters
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dropout-mode', default='random_drop')
    parser.add_argument('--dropout-begin-step', type=float, default=0.0)
    parser.add_argument('--dropout-end-step', type=float, default=20000.0)
    parser.add_argument('-t', '--temp', type=float, default=1.)
    parser.add_argument('-l', '--lamb', type=float, default=.8)
    parser.add_argument('--lr', type=float, default=1e-2)
    
    return parser.parse_args()

def rate_fn(step, begin_step, end_step):
    total_step = end_step - begin_step
    rate = tf.clip_by_value((step - begin_step) / total_step, 0., 1.)
    
    return rate

class TeacherModel(tf.keras.Model):
    def __init__(self, num_class):
        super(TeacherModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.d2 = tf.keras.layers.Dense(32, activation='relu')
        self.d3 = tf.keras.layers.Dense(32, activation='relu')
        self.d4 = tf.keras.layers.Dense(num_class)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return tf.keras.activations.softmax(x)

    def get_logits(self, x, temp=1.):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return tf.keras.activations.softmax(tf.math.divide(x, temp))

class StudentModel(tf.keras.Model):
    def __init__(self, dropout_dims, dropout_rate_fn, dropout_mode, num_class, temp):
        super(StudentModel, self).__init__()
        self.d1 = DropoutDense(32, activation='relu',
                               dropout_dims=dropout_dims,
                               dropout_rate_fn=dropout_rate_fn,
                               dropout_mode=dropout_mode)
        self.d2 = tf.keras.layers.Dense(32, activation='relu')
        self.d3 = tf.keras.layers.Dense(32, activation='relu')
        self.d4 = tf.keras.layers.Dense(num_class)

        self.temp = temp

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        p = tf.keras.activations.softmax(x)
        pt = tf.keras.activations.softmax(tf.math.divide(x, self.temp))
        return p, pt

def run_once(args):
    ### config
    num_class = NUM_CLASS[args.dataset]
    if args.miss_type == 'random':
        miss_dims = args.miss_ratio
    elif args.miss_type == 'config':
        miss_dims = MISS_CONFIG[args.dataset][args.miss_config]
    else:
        raise NotImplementedError

    ### data
    dp = get_data_processor(args.dataset)
    x_tr, y_tr, x_ts, y_ts = dp.get_full()
    miss_indicator, _ = dp.get_miss_indicator(miss_dims=miss_dims)
    
    ### teacher
    model_teacher = TeacherModel(num_class)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    model_teacher.compile(optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    # train & evaluate
    model_teacher.fit(x_tr, y_tr,
        epochs=args.epochs * 2, batch_size=args.batch_size)
    _, acc_teacher = model_teacher.evaluate(x_ts, y_ts, verbose=2)

    ### infer logits from teacher
    logits_tr = model_teacher.get_logits(x_tr, args.temp).numpy()
    logits_ts = model_teacher.get_logits(x_ts, args.temp).numpy()

    ### dropout rate function
    dropout_rate_fn = partial(rate_fn,
        begin_step=args.dropout_begin_step, end_step=args.dropout_end_step)

    ### Student
    model_student = StudentModel(dropout_dims=miss_indicator,
        dropout_rate_fn=dropout_rate_fn, dropout_mode=args.dropout_mode,
        num_class=num_class, temp=args.temp)
    
    # decay
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    model_student.compile(optimizer=optimizer,
        loss=['sparse_categorical_crossentropy', 'categorical_crossentropy'],
        metrics=[['sparse_categorical_accuracy'], ['categorical_crossentropy']],
        loss_weights=[1., 0.])
    model_student.fit(x_tr, [y_tr, logits_tr],
        epochs=args.epochs * 2, batch_size=args.batch_size)
    # evaluate
    _, _, _, acc_student_decay, _ = model_student.evaluate(x_ts, [y_ts, logits_ts], verbose=2)
    
    # distillation
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    model_student.compile(optimizer=optimizer,
        loss=['sparse_categorical_crossentropy', 'categorical_crossentropy'],
        metrics=[['sparse_categorical_accuracy'], ['categorical_crossentropy']],
        loss_weights=[args.lamb, 1.])
    model_student.fit(x_tr, [y_tr, logits_tr],
        epochs=args.epochs, batch_size=args.batch_size)
    # evaluate
    _, _, _, acc_student_distill, _ = model_student.evaluate(x_ts, [y_ts, logits_ts], verbose=2)
    
    return acc_teacher, acc_student_decay, acc_student_distill


args = parse_args()
exp_name = '-'.join([args.dataset, args.miss_type, 'decay_distill'])
result_file_path = os.path.join('result', exp_name + '.txt')

AT_list = []            # AT = Accuracy of Teacher
ASDecay_list = []       # AS = Accuracy of Student
ASDistill_list = []     # AS = Accuracy of Student
for i in range(args.num_exp):
    acc_teacher, acc_student_decay, acc_student_distill = run_once(args)
    AT_list.append(acc_teacher)
    ASDecay_list.append(acc_student_decay)
    ASDistill_list.append(acc_student_distill)
AT_mean, AT_std = np.mean(AT_list), np.std(AT_list)
ASDecay_mean, ASDecay_std = np.mean(ASDecay_list), np.std(ASDecay_list)
ASDistill_mean, ASDistill_std = np.mean(ASDistill_list), np.std(ASDistill_list)

with open(result_file_path, 'a') as f:
    f.write('timestamp: {:s}\n'.format(time.asctime()))
    f.write('data setting:\n')
    f.write('  dataset: {:s}\n'.format(args.dataset))
    f.write('  miss-type: {:s}\n'.format(args.miss_type))
    if args.miss_type == 'random':
        f.write('  miss-ratio: {:f}\n'.format(args.miss_ratio))
    elif args.miss_type == 'config':
        f.write('  miss-config: {:s}\n'.format(args.miss_config))
    else:
        raise NotImplementedError
    f.write('experiment setting:\n')
    f.write('  num-exp: {:d}\n'.format(args.num_exp))
    f.write('training parameters:\n')
    f.write('  epochs: {:d}\n'.format(args.epochs))
    f.write('  batch-size: {:d}\n'.format(args.batch_size))
    f.write('  dropout-mode: {:s}\n'.format(args.dropout_mode))
    f.write('  dropout-begin-step: {:.1f}\n'.format(args.dropout_begin_step))
    f.write('  dropout-end-step: {:.1f}\n'.format(args.dropout_end_step))
    f.write('  temp: {:f}\n'.format(args.temp))
    f.write('  lamb: {:f}\n'.format(args.lamb))
    f.write('  lr: {:f}\n'.format(args.lr))
    f.write('experiment result:\n')
    f.write('  teacher-acc: {:.8f}/{:.8f}\n'.format(AT_mean, AT_std))
    f.write('  decay-acc:   {:.8f}/{:.8f}\n'.format(ASDecay_mean, ASDecay_std))
    f.write('  distill-acc: {:.8f}/{:.8f}\n\n'.format(ASDistill_mean, ASDistill_std))
