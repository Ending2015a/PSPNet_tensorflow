import os
import glob

def get_dir_list(dir):
    return [f in os.listdir(dir) if os.isdir(f)]

def get_file_list(dir, ext='*'):
    if not ext == '*':
        ext = '*.' + ext
    path = os.path.join(dir, ext)
    return glob.glob(path)

def get_list(dir, ext='*'):
    return get_dir_list(dir), get_file_list(dir, ext)

name = input('list name:')
root = input('root folder:')
image = input('image folder (from root):')
anno = input('anno folder (from root):')

if root == '':
    root = os.path.abspath(__file__)

if not os.path.isdir(root):
    print('invalid root folder path.')
    exit()

image_path = os.path.join(root, image)
anno_path = os.path.join(root, anno)



