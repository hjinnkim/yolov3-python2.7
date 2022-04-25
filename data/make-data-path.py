import os
import glob

def run():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = base_dir+"/custom"
    images_dir = dataset_dir+'/images'
    img_files = sorted(glob.glob(images_dir+"/*"))

    traintxt = os.path.join(dataset_dir, 'train.txt')
    validtxt = os.path.join(dataset_dir, 'valid.txt')

    with open(traintxt, 'w') as f:
        for img_file in img_files:
            f.write(img_file+'\n')

    with open(validtxt, 'w') as f:
        for img_file in img_files:
            f.write(img_file+'\n')


if __name__=="__main__":
    run()