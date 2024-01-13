import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == '__main__':

    data_path = './data/'
    output_path = './'
    num_augmentations = 20
    train_size = 0.8
    with_flip = True

    path_temp = "E:/teeth_3D/3DTeethSeg22_challenge/teeth_3D_data_center_trans/teeth/"
    file_lists = os.listdir(path_temp)
    print(file_lists)

    sample_list = np.asarray(sample_list)


    train_list, val_list = train_test_split(sample_list, train_size=0.8, shuffle=True)
    val_list, test_list = train_test_split(val_list, train_size=0.5, shuffle=True)
    print('Training list:\n', train_list, '\nValidation list:\n', val_list, '\nTest list:\n', test_list)

    #training
    train_name_list = []
    for i_sample in train_list:
        #print('Computing Sample: {0}; Aug: {1}...'.format(i_sample, i_aug))
        subject_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample)
        train_name_list.append(os.path.join(data_path, subject_name))
        if with_flip:
            subject2_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample+1000)
            train_name_list.append(os.path.join(data_path, subject2_name))

    with open(os.path.join(output_path, 'train_list_{0}.csv'.format(i_cv)), 'w') as file:
        for f in train_name_list:
            file.write(f+'\n')


    print('--------------------------------------------')
    print('with flipped samples:', with_flip)
    print('# of train:', len(train_name_list))
    print('# of validation:', len(val_name_list))
    print('--------------------------------------------')


    #print(sample_list)

    i_cv = 0
    kf = KFold(n_splits=6, shuffle=False)
    for train_idx, test_idx in kf.split(sample_list):

        i_cv += 1
        print('Round:', i_cv)

        train_list, test_list = sample_list[train_idx], sample_list[test_idx]
        train_list, val_list = train_test_split(train_list, train_size=0.8, shuffle=True)

        print('Training list:\n', train_list, '\nValidation list:\n', val_list, '\nTest list:\n', test_list)

        #training
        train_name_list = []
        for i_sample in train_list:
            for i_aug in range(num_augmentations):
                #print('Computing Sample: {0}; Aug: {1}...'.format(i_sample, i_aug))
                subject_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample)
                train_name_list.append(os.path.join(data_path, subject_name))
                if with_flip:
                    subject2_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample+1000)
                    train_name_list.append(os.path.join(data_path, subject2_name))

        with open(os.path.join(output_path, 'train_list_{0}.csv'.format(i_cv)), 'w') as file:
            for f in train_name_list:
                file.write(f+'\n')

        #validation
        val_name_list = []
        for i_sample in val_list:
            for i_aug in range(num_augmentations):
                #print('Computing Sample: {0}; Aug: {1}...'.format(i_sample, i_aug))
                subject_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample)
                val_name_list.append(os.path.join(data_path, subject_name))
                if with_flip:
                    subject2_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample+1000)
                    val_name_list.append(os.path.join(data_path, subject2_name))

        with open(os.path.join(output_path, 'val_list_{0}.csv'.format(i_cv)), 'w') as file:
            for f in val_name_list:
                file.write(f+'\n')

        #test
        test_df = pd.DataFrame(data=test_list, columns=['Test ID'])
        test_df.to_csv('test_list_{}.csv'.format(i_cv), index=False)


        print('--------------------------------------------')
        print('with flipped samples:', with_flip)
        print('# of train:', len(train_name_list))
        print('# of validation:', len(val_name_list))
        print('--------------------------------------------')
