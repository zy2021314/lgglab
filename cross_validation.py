import numpy as np
import datetime
import os
import csv
import h5py
import copy
import os.path as osp
from train_model import *
from utils import Averager, ensure_path
from sklearn.model_selection import KFold
import pickle
from struct import unpack
unpack
ROOT = os.getcwd()

"""
交叉验证
"""

class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        result_path = osp.join(args.save_path, 'result')
        ensure_path(result_path)
        self.text_file = osp.join(result_path,
                                  "results_{}.txt".format(args.dataset))
        file = open(self.text_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                   "\n1)number_class:" + str(args.num_class) +
                   "\n2)random_seed:" + str(args.random_seed) +
                   "\n3)learning_rate:" + str(args.learning_rate) +
                   "\n4)pool:" + str(args.pool) +
                   "\n5)num_epochs:" + str(args.max_epoch) +
                   "\n6)batch_size:" + str(args.batch_size) +
                   "\n7)dropout:" + str(args.dropout) +
                   "\n8)hidden_node:" + str(args.hidden) +
                   "\n9)input_shape:" + str(args.input_shape) +
                   "\n10)class:" + str(args.label_type) +
                   "\n11)T:" + str(args.T) +
                   "\n12)graph-type:" + str(args.graph_type) + '\n')
        file.close()

    def load_per_subject(self, sub):
        """
        load data for sub
        将刚刚生成的数据读取
        :param sub: which subject's data to load
        :return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = osp.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    def prepare_data(self, idx_train, idx_test, data, label):

        #只拿其中设置的数据进行划分
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]

        # if self.args.dataset == 'Att' or self.args.dataset == 'DEAP':
        #     """
        #     For DEAP we want to do trial-wise 10-fold, so the idx_train/idx_test is for
        #     trials.
        #     data: (trial, segment, 1, chan, datapoint)
        #     To use the normalization function, we should change the dimension from
        #     (trial, segment, 1, chan, datapoint) to (trial*segments, 1, chan, datapoint)
        #     我们希望进行10次试验，因此idx_train/idx_test用于训练
        #     数据:(trial, segment, 1, chan, datpoint)
        #     为了使用归一化函数，我们应该改变维数
        #     (trial, segment, 1, chan, datapopoint)到(trial*segments, 1, chan, datapopoint)
        #     """
        #
        #     #（40,15）-》（40*15）
        #     data_train = np.concatenate(data_train, axis=0)
        #     label_train = np.concatenate(label_train, axis=0)
        #     if len(data_test.shape) > 4:
        #         """
        #         When leave one trial out is conducted, the test data will be (segments, 1, chan, datapoint), hence,
        #         no need to concatenate the first dimension to get trial*segments
        #         当进行一次试验时，测试数据为(segments, 1, chan, datpoint)，因此，不需要连接第一个维度来获得试验*段
        #         """
        #         data_test = np.concatenate(data_test, axis=0)
        #         label_test = np.concatenate(label_test, axis=0)

        data_train, data_test = self.normalize(train=data_train, test=data_test)
        # Prepare the data format for training the model using PyTorch
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test

    def normalize(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        该函数对脑电通道逐个通道进行标准归一化处理
        :param train: training data (sample, 1, chan, datapoint)
        :param test: testing data (sample, 1, chan, datapoint)
        :return: normalized training and testing data
        """
        # data: sample x 1 x channel x data data：（540,1,34,512）
        for channel in range(train.shape[2]):
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
            test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        """
        Get the validation set using the same percentage of the two classe samples
        使用两个类样本的相同百分比获得验证集
        :param data: training data (segment, 1, channel, data)
        :param label: (segments,)
        :param train_rate: the percentage of trianing data
        :param random: bool, whether to shuffle the training data before get the validation data
        :return: data_trian, label_train, and data_val, label_val
        """
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        if random == True:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label

    #
    def n_fold_CV(self, subject=[0], fold=10, shuffle=True):
        """
        this function achieves n-fold cross-validation
        该函数实现了n次交叉验证
        :param subject: how many subject to load
        :param fold: how many fold
        """
        # Train and evaluate the model subject by subject
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1

        for sub in subject:
            #读取，hdf文件
            data, label = self.load_per_subject(sub)

            va_val = Averager()
            vf_val = Averager()
            preds, acts = [], []
            #创建一个kf变量
            kf = KFold(n_splits=fold, shuffle=shuffle)

            #split(X, y=None, groups=None)：将数据集划分成训练集和测试集，返回索引生成器

            for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
                #准备数据,一个人是（40,15,1,34，512），train：540，test：60.labels：（60,1）
                data_train, label_train, data_test, label_test = self.prepare_data(
                    idx_train=idx_train, idx_test=idx_test, data=data, label=label)

                if self.args.reproduce:
                    # to reproduce the reported ACC来重现报告的ACC
                    acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)
                    acc_val = 0
                    f1_val = 0
                else:
                    # to train new models
                    acc_val, f1_val = self.first_stage(data=data_train, label=label_train,
                                                      subject=sub, fold=idx_fold)

                    combine_train(args=self.args,
                                  data=data_train, label=label_train,
                                  subject=sub, fold=idx_fold, target_acc=1)

                    acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)
                va_val.add(acc_val)
                vf_val.add(f1_val)
                preds.extend(pred)
                acts.extend(act)

            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc)
            ttf.append(f1)
            result = '{},{}'.format(tta[-1], f1)
            self.log2txt(result)



        # prepare final report
        tta = np.array(tta)
        ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        std = np.std(tta)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{}'.format(mF1_val))
        results = 'test mAcc={} mF1={} val mAcc={} val F1={}'.format(mACC,
        mF1, mACC_val, mF1_val)
        self.log2txt(results)


    def first_stage(self, data, label, subject, fold):
        """
        this function achieves n-fold-CV to:
            1. select hyper-parameters on training data
            2. get the model for evaluation on testing data
            此函数实现n-fold-CV为:
            1. 选择训练数据的超参数
            2. 得到对测试数据进行评估的模型
        :param data: (segments, 1, channel, data)
        :param label: (segments,)
        :param subject: which subject the data belongs to
        :param fold: which fold the data belongs to
        :return: mean validation accuracy平均验证集准确率
        """
        # use n-fold-CV to select hyper-parameters on training data使用n-fold cv选择训练数据的超参数
        # save the best performance model and the corresponding acc for the second stage 为第二阶段保存最佳性能模型和相应的acc
        # data: trial x 1 x channel x time
        kf = KFold(n_splits=3, shuffle=True)

        va = Averager()
        vf = Averager()
        va_item = []
        maxAcc = 0.0
        for i, (idx_train, idx_val) in enumerate(kf.split(data)):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            #data：540，train：360，val：180
            data_train, label_train = data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]
            acc_val, F1_val = train(args=self.args,
                                    data_train=data_train,
                                    label_train=label_train,
                                    data_val=data_val,
                                    label_val=label_val,
                                    subject=subject,
                                    fold=fold)
            va.add(acc_val)
            vf.add(F1_val)
            va_item.append(acc_val)

            if acc_val >= maxAcc:
                maxAcc = acc_val
                # choose the model with higher val acc as the model to second stage  第二阶段模型选择val acc较高的模型
                old_name = osp.join(self.args.save_path, 'candidate.pth')
                new_name = osp.join(self.args.save_path, 'max-acc.pth')
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
                print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))

        mAcc = va.item()
        mF1 = vf.item()
        return mAcc, mF1

    def log2txt(self, content):
        """
        this function log the content to results.txt
        此函数将内容记录到results.txt
        :param content: string, the content to log
        """
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()

