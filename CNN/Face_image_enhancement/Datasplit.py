import os
import shutil

def make_dir(train_path, test_path):
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
        print('make a train directory')
    if not os.path.isdir(test_path):
        os.mkdir(test_path)
        print('make a test directory')

def split_dataset(d_path, train_path, test_path):

    #load all folder directory
    face_list = os.listdir(os.path.join(d_path, 'lfw_funneled'))
    #print(face_list)
    for face in face_list:
        if face[-4:] =='.txt': continue
        src = os.path.join(d_path,'lfw_funneled',face)
        print(src)
        if face[0] == 'X' or face[0] == 'Y' or face[0] =='Z':
            dst = os.path.join(test_path, face)
        else:
            dst = os.path.join(train_path, face)

        # ignore the exist directory
        if os.path.isdir(dst):
            continue
        shutil.copytree(src, dst)
    print('split complete')


if __name__ == '__main__':
    d_path = './dataset'

    # make train, test folder
    train_path = os.path.join(d_path, 'train')
    test_path = os.path.join(d_path, 'test')
    make_dir(train_path, test_path)

    # split dataset
    split_dataset(d_path, train_path, test_path)
