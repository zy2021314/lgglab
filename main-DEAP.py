from cross_validation import *
from prepare_data_DEAP import *
import argparse

if __name__ == '__main__':
    #定义命令行参数的模块
    parser = argparse.ArgumentParser()

    ######## Data ########
    """
    LAB对应datasetman
    DEAP对应data_preprocessed_python
    HCI对应Sessions
    可以先运行准备数据，来看输入形状
    # pd = PrepareData(args)
    # pd.run(sub_to_run, split=True, expand=True)
    """
    parser.add_argument('--dataset', type=str, default='LAB', choices=['DEAP', 'HCI', 'LAB', 'SEED'])
    parser.add_argument('--data-path', type=str, default='dataset_man/', choices=['Sessions/', 'data_preprocessed_python/', 'dataset_man/'])
    parser.add_argument('--input-shape', type=tuple, default=(1, 14, 128))
    ################## 更改数据时需要将上面三者都更改##########

    #定义人数
    parser.add_argument('--subjects', type=int, default=1)

    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='V', choices=['A', 'V', 'D', 'L'])

    ##########分片的参数################################
    parser.add_argument('--segment', type=int, default=1)#每一片的时间长短
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--scale-coefficient', type=float, default=1)
    ###############################

    parser.add_argument('--data-format', type=str, default='eeg')
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=7)
    parser.add_argument('--max-epoch', type=int, default=200)
    #patient，表示，第一阶段训练时，能忍受的训练集精度为1的次数
    #max-epoch-cmb表示，第二阶段训练的epoch
    parser.add_argument('--patient', type=int, default=20)
    parser.add_argument('--patient-cmb', type=int, default=8)
    parser.add_argument('--max-epoch-cmb', type=int, default=20)#第二阶段训练的epoch
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--LS', type=bool, default=True, help="Label smoothing")
    parser.add_argument('--LS-rate', type=float, default=0.1)
    ##########################模型保存路径，如果要切换保存路径，在运行前切换即可###############################
    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--load-path-final', default='./save/final_model.pth')
    #################################################################
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='VIT')
    parser.add_argument('--pool', type=int, default=16)
    parser.add_argument('--pool-step-rate', type=float, default=0.25)
    parser.add_argument('--T', type=int, default=64)
    #
    parser.add_argument('--graph-type', type=str, default='TS', choices=['fro', 'gen', 'hem', 'BL', 'TS', 'TS2', 'hem2'])
    parser.add_argument('--hidden', type=int, default=32)

    ########对于vit的参数 ######
    parser.add_argument('--depth', type=int, default=1)#transform模块数量
    ######## Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', type=bool, default=False)
    args = parser.parse_args()


    sub_to_run = np.arange(args.subjects)#[0,1----]
    if args.dataset == 'HCI' or args.dataset == 'LAB':
            #因为hci数据中 id 从1开始
            sub_to_run = [i+1 for i in sub_to_run]


    pd = PrepareData(args)
    pd.run(sub_to_run, split=True, expand=True)
    cv = CrossValidation(args)
    seed_all(args.random_seed)
    cv.n_fold_CV(subject=sub_to_run, shuffle=True, fold=10)
    #cv.indepent_train(subject=sub_to_run)
    #cv.norm_train(sub_to_run)
    #在cv中添加函数，构造新的训练方式
