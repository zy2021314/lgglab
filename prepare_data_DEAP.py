# This is the processing script of DEAP dataset

import _pickle as cPickle
import xml.dom.minidom
from xml.dom import Node
import numpy as np
import pandas as pd
from pyedflib import highlevel
import re
import os
from train_model import *
from scipy import signal


class PrepareData:
    def __init__(self, args):
        # init all the parameters here # init这里的所有参数
        # arg contains parameter settings arg包含参数设置
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type
        if self.args.dataset =='DEAP':
            #可以自己按照索引确定需要的信道和个数
            self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                                   'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
                                   'CP2', 'P4', 'P8', 'PO4', 'O2', 'eog1', 'eog2','q1','q2','q3','q4','q5','q6']
            self.graph_fro_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                                   ['Fz'],
                                   ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                                   ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                                   ['T7'], ['T8'], ['eog1', 'eog2']]#添加EOG信号眼电信号
            self.graph_gen_DEAP = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                                   ['FC5', 'FC1', 'FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                                   ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                                   ['T7'], ['T8'], ['eog1','eog2','q1','q2','q3','q4','q5','q6']]
            self.graph_hem_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                                   ['Fz', 'Cz', 'Pz', 'Oz'],
                                   ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3'], ['C4'], ['CP5', 'CP1'], ['CP2', 'CP6'],
                                   ['P7', 'P3'], ['P4', 'P8'], ['PO3', 'O1'], ['PO4', 'O2'], ['T7'], ['T8']]
            #添加EXG信号
            self.TS = [['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3','O1'],
                       ['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
                       ['eog1','eog2','q1','q2','q3','q4','q5','q6']]


        elif self.args.dataset == 'HCI':
            self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                                   'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
                                   'CP2', 'P4', 'P8', 'PO4', 'O2', 'EXG1', 'EXG2', 'EXG3','EXG4','EXG5','EXG6','EXG7','EXG8',
                                   'GSR1', 'GSR2', 'ERG1', 'ERG2', 'RESP', 'TEMP', 'STATUS']
            #添加EXG信号
            #不变channel and 多模态
            self.TS = [['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3','O1', 'Oz', 'Pz'],
                       ['Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6','CP2', 'P4', 'P8', 'PO4', 'O2'],
                       ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'ERG1', 'ERG2', 'RESP', 'TEMP', 'STATUS']]
            #变channel and 多模态
            self.graph_hem_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                                   ['Fz', 'Cz', 'Pz', 'Oz'],
                                   ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3'], ['C4'], ['CP5', 'CP1'], ['CP2', 'CP6'],
                                   ['P7', 'P3'], ['P4', 'P8'], ['PO3', 'O1'], ['PO4', 'O2'], ['T7'], ['T8'],
                                    ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'ERG1', 'ERG2', 'RESP', 'TEMP', 'STATUS']]
            # 变channel and no多模态
            self.graph_hem_DEAP2 = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                                   ['Fz', 'Cz', 'Pz', 'Oz'],
                                   ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3'], ['C4'], ['CP5', 'CP1'], ['CP2', 'CP6'],
                                   ['P7', 'P3'], ['P4', 'P8'], ['PO3', 'O1'], ['PO4', 'O2'], ['T7'], ['T8'],
                                    ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'ERG1', 'ERG2', 'RESP', 'TEMP', 'STATUS']]
            #不变channel and no多模态
            self.TS2 = [
                ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz'],
                ['Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']]

        # 添加，其他的数据集

        self.graph_type = args.graph_type

    def run(self, subject_list, split=False, expand=True):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment是否将一个试验的数据分割为更短的段
        expand: (bool) whether to add an empty dimension for CNN 是否为CNN添加一个空维度

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        if self.args.dataset == 'DEAP':
            for sub in subject_list:
                data_, label_ = self.load_data_per_subject(sub)
                # select label type here
                label_ = self.label_selection(label_)

                if expand:
                    # expand one dimension for deep learning(CNNs)
                    #（40,32,7680）-》》（40,1,32,7680）
                    data_ = np.expand_dims(data_, axis=-3)

                if split:
                    data_, label_ = self.split(
                                        data=data_,
                                        label=label_,
                                        segment_length=self.args.segment,
                                        overlap=self.args.overlap, sampling_rate=self.args.sampling_rate
                                            )

                print('Data and label prepared!')
                print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
                print('----------------------')
                self.save(data_, label_, sub)
        elif self.args.dataset == 'HCI':
            #因为hci数据中 id 从1开始
            base_path  = self.args.data_path
            folder_name = os.listdir(base_path)
            session_path = os.path.join(base_path, folder_name[1], 'session.xml')
            for sub in subject_list:
                #对于每一个受试者需要将其分片后存入数据中
                data_ = []
                label_ = []
                for fold in folder_name:
                    #print(fold)
                    folder_path = os.path.join(base_path, fold)
                    file_name = os.listdir(folder_path)
                    session_path = os.path.join(folder_path, 'session.xml')
                    dom = xml.dom.minidom.parse(session_path)
                    root = dom.documentElement
                    for i in file_name:
                        if os.path.splitext(i)[1] == ".bdf":
                            #print(float(root.getAttribute('feltVlnc')))
                            label_one_movie = int(float(root.getAttribute('feltVlnc')))
                            if int(root.getElementsByTagName('subject')[0].getAttribute('id')) == sub:
                                #确定了文件为这个受试者的数据,不知道为什么在if里面root就有问题了
                                #标签二分类
                                print(fold)
                                if self.args.num_class == 2:
                                    label_one_movie = np.where(label_one_movie <= 5, 0, label_one_movie)
                                    label_one_movie = np.where(label_one_movie > 5, 1, label_one_movie)
                                #print(label_one_movie)
                                #读取数据
                                for i in file_name:
                                    if os.path.splitext(i)[1] == ".bdf":
                                        eeg_file_path = os.path.join(folder_path, i)
                                        data_one_movie, _, _ = highlevel.read_edf(eeg_file_path)
                                #得到其中一个的数据后，将其扩充维度，分片并且标准化，最后在最高维度合并后存入hdf文件
                                if expand:
                                    data_one_movie = np.expand_dims(data_one_movie, axis=-3)
                                data_one_movie, label_one_movie = self.hci_split(
                                    data=data_one_movie,
                                    label=label_one_movie,
                                    segment_length=self.args.segment,
                                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate
                                )
                                data_.append(data_one_movie)
                                label_.append(label_one_movie)


                data_ = np.concatenate(data_, axis=0)
                label_ = np.concatenate(label_, axis=0)
                data_ = self.hci_reorder_channel(data=data_, graph=self.graph_type)
                print("The one people Data shape:" + str(data_.shape) + " Label:" + str(
                    label_.shape))
                self.save(data_, label_, sub)

        elif self.args.dataset == 'LAB':
            #base_path = self.args.data_path
            folder_name = os.listdir('dataset_man')
            folder_len = len(os.listdir('dataset_man'))

            for sub in subject_list:
                label = []
                data = []
                for i in folder_name:
                    if int(re.sub("\D", "", i)) == sub:
                        label_name = re.search('[^/_]+$', i).group()
                        file_name = os.listdir(f'dataset_man/{i}')
                        file_len = len(os.listdir(f'dataset_man/{i}'))
                        # print(label_name)
                        if label_name == 'like':
                            label.append(np.ones(file_len))
                        else:
                            label.append(np.zeros(file_len))
                        for j in file_name:
                            temp = pd.read_csv(f"dataset_man/{i}/{j}", index_col=0)
                            data.append(temp)

                data = np.array(data)
                data = data[:, :, :256]
                #
                label = [j for i in label for j in i]
                label = [int(i) for i in label]
                label = np.array(label)
                if expand:
                    data = np.expand_dims(data, axis=-3)
                data, label = self.split(
                    data=data,
                    label=label,
                    segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate
                )
                self.save(data, label, sub)
        elif self.args.dataset == 'SHOP':
            # label == Disike or Like
            # 数据集特点：一个人数量不一定为42个，命名规范为：name_序号,data和label的命名规范一致
            # 方法1：在遇到该实验为空的时候跳过
            # 方法2，根据命名规范，设置不同的list
            # 由于两个都为乱序，所以我的思路为，提取实验者姓名，提取实验编号，作为变量
            # for循环遍历文件名，如果开头几个字符等于姓名，则继续操作（会出错）,加一个限制条件， and 整个文件名小于开头字符+6，即可
            # 思考，能不能乱序
            base_datapath = 'Data-EEG-25-users-Neuromarketing/25-users'
            base_labelpath = 'Data-EEG-25-users-Neuromarketing/labels'
            data_name = os.listdir('Data-EEG-25-users-Neuromarketing/25-users')
            label_name = os.listdir('Data-EEG-25-users-Neuromarketing/labels')
            data_name.sort()
            label_name.sort()
            names = []
            nums = []
            for i in range(len(data_name)):
                name = re.search("(.*?)(\d*.txt)", data_name[i]).group(1)
                num = re.search("(.*?)(\d*.txt)", data_name[i]).group(2)
                names.append(name)
                nums.append(num)
            #25个人的名字
            sub_name = list(set(names))
            sub_name.sort()
            for sub in subject_list:
                flag = 0
                data = []
                label = []
                for i in range(len(data_name)):
                    # 获得名字长度
                    if flag == 42:
                        break
                    name_len = len(sub_name[sub])
                    # 因为名字有这样子的Gautam_  | Gautam_123_ 所以加上第二个限制条件
                    if data_name[i][:name_len] == sub_name[sub] and len(data_name[i]) <= (name_len+7):
                        #读取数据 和 标签
                        flag = flag + 1
                        data_path = os.path.join(base_datapath, data_name[i])
                        label_path = os.path.join(base_labelpath, label_name[i])
                        #label_path = "Data-EEG-25-users-Neuromarketing/labels/" + label_name[i]
                        data.append(np.loadtxt(data_path).T)
                        f = open(label_path)
                        label_str = f.read()
                        if label_str == 'Like':
                            label.append(int(1))
                        #够不严谨的，dislike都能拼错
                        elif label_str == 'Disike':
                            label.append(int(0))


                data = np.array(data)
                # data = data[:, :, :256]
                # label = [j for i in label for j in i]
                label = [int(i) for i in label]
                label = np.array(label)
                if expand:
                    data = np.expand_dims(data, axis=-3)
                data, label = self.split(
                    data=data,
                    label=label,
                    segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate
                )
                self.save(data, label, sub)









    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        这个函数加载目标人物的原始文件
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: (40, 32, 7680) label: (40, 4)
        """
        sub += 1
        #if只是为了匹配相应的文件名
        if (sub < 10):
            sub_code = str('s0' + str(sub) + '.dat')
        else:
            sub_code = str('s' + str(sub) + '.dat')
        subject_path = os.path.join(self.data_path, sub_code)

        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        label = subject['labels']
        #data = subject['data'][:, 0:32, 3 * 128:]  # Excluding the first 3s of baseline
        #添加自身其他信道的点
        data = subject['data'][:, :, 3 * 128:]
        #EXG_data = subject['data'][:, 33:34, 3 * 128:]
        #   data: 40 x 32 x 7680
        #   label: 40 x 4
        # reorder the EEG channel to build the local-global graphs #重新排序脑电通道，构建局部-全局图
        data = self.reorder_channel(data=data, graph=self.graph_type)
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label

    def load_hci_data_per_subject(self, sub):
        """

        """

    def reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        该功能根据不同的图形设计对通道进行重新排序
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'fro':
            graph_idx = self.graph_fro_DEAP
        elif graph == 'gen':
            graph_idx = self.graph_gen_DEAP
        elif graph == 'hem':
            graph_idx = self.graph_hem_DEAP
        elif graph == 'BL':
            graph_idx = self.original_order
        elif graph == 'TS':
            graph_idx = self.TS
        #这一步的目的是将data变为相应的形状
        idx = []
        if graph in ['BL']:
            for chan in graph_idx:
                idx.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):#grath-idx=[['eog1','eog2','q1','q2','q3','q4','q5','q6']]
                num_chan_local_graph.append(len(graph_idx[i]))#将每一个小元组的个数输入ex：【1,2,2,2,】
                for chan in graph_idx[i]:#对每个元素取其中索引加入索引列表，例如像上面的元素，取值就取，32----40
                    idx.append(self.original_order.index(chan))

            # save the number of channels in local graph for building the LGG model in utils.py
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
        return data[:, idx, :]

    def hci_reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        该功能根据不同的图形设计对通道进行重新排序
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'fro':
            graph_idx = self.graph_fro_DEAP
        elif graph == 'gen':
            graph_idx = self.graph_gen_DEAP
        elif graph == 'hem':
            graph_idx = self.graph_hem_DEAP
        elif graph == 'BL':
            graph_idx = self.original_order
        elif graph == 'TS':
            graph_idx = self.TS
        #这一步的目的是将data变为相应的形状
        idx = []
        if graph in ['BL']:
            for chan in graph_idx:
                idx.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):#grath-idx=[['eog1','eog2','q1','q2','q3','q4','q5','q6']]
                num_chan_local_graph.append(len(graph_idx[i]))#将每一个小元组的个数输入ex：【1,2,2,2,】
                for chan in graph_idx[i]:#对每个元素取其中索引加入索引列表，例如像上面的元素，取值就取，32----40
                    idx.append(self.original_order.index(chan))

            # save the number of channels in local graph for building the LGG model in utils.py
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
        return data[:, :, idx, :]

    def label_selection(self, label):
        """
        This function: 1. selects which dimension of labels to use
                       2. create binary label
        Parameters
        ----------
        label: (trial, 4)

        Returns
        -------
        label: (trial,)
        """
        if self.label_type == 'A':
            label = label[:, 1]
        elif self.label_type == 'V':
            label = label[:, 0]
        elif self.label_type == 'D':
            label = label[:, 2]
        elif self.label_type == 'L':
            label = label[:, 3]
        if self.args.num_class == 2:
            label = np.where(label <= 5, 0, label)
            label = np.where(label > 5, 1, label)
            print('Binary label generated!')
        return label

    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    def split(self, data, label, segment_length=1, overlap=0, sampling_rate=256):
        """
        This function split one trial's data into shorter segments片段
        该函数将一次试验的数据分成更短的段片段
        分割为15断，每段4秒
        Parameters
        ----------
        data: (trial, f, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, f, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []
        #
        number_segment = int((data_shape[-1] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, :, (i * step):(i * step + data_segment)])

        data_split_array = np.stack(data_split, axis=1)
        #label=（40，） ---》（40，片段数
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label

    def hci_split(self, data, label, segment_length=1, overlap=0, sampling_rate=256):
        """
        This function split one trial's data into shorter segments片段
        该函数将一次试验的数据分成更短的段片段
        分割为15断，每段4秒
        Parameters
        ----------
        data: (trial, f, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, f, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        #每一步的步长为时间长度*采样率*重叠率
        step = int(segment_length * sampling_rate * (1 - overlap))
        #每一个片段长度
        data_segment = sampling_rate * segment_length
        data_split = []
        #片段个数为
        number_segment = int((data_shape[-1] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, (i * step):(i * step + data_segment)])

        data_split_array = np.stack(data_split, axis=0)
        label = np.repeat(label, int(number_segment + 1))
        #label = np.stack(np.repeat(label[i], int(number_segment + 1)), axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label