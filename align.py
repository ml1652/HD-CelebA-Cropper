from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
from multiprocessing import Pool
import os
import re

import cropper
import numpy as np
import tqdm


# ==============================================================================
# =                                      param                                 =
# ==============================================================================

parser = argparse.ArgumentParser()
# main
parser.add_argument('--img_dir', dest='img_dir', default='./data/img_celeba')
parser.add_argument('--save_dir', dest='save_dir', default='./data/aligned')
parser.add_argument('--landmark_file', dest='landmark_file', default='./data/landmark.txt')
parser.add_argument('--standard_landmark_file', dest='standard_landmark_file', default='./data/standard_landmark_68pts.txt')
parser.add_argument('--crop_size_h', dest='crop_size_h', type=int, default=572)
parser.add_argument('--crop_size_w', dest='crop_size_w', type=int, default=572)
parser.add_argument('--move_h', dest='move_h', type=float, default=0.25)
parser.add_argument('--move_w', dest='move_w', type=float, default=0.)
parser.add_argument('--save_format', dest='save_format', choices=['jpg', 'png'], default='jpg')
parser.add_argument('--n_worker', dest='n_worker', type=int, default=4)
# others
parser.add_argument('--face_factor', dest='face_factor', type=float, help='The factor of face area relative to the output image.', default=0.45) #default = 0.5
parser.add_argument('--align_type', dest='align_type', choices=['affine', 'similarity'], default='similarity')
parser.add_argument('--order', dest='order', type=int, choices=[0, 1, 2, 3, 4, 5], help='The order of interpolation.', default=3)
parser.add_argument('--mode', dest='mode', choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'], default='edge')
args = parser.parse_args()

def draw_face_landmark(name, landmarkToPaint, img_crop):
    # draw face's landmark
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8
    name_landmark_str = name
    for point in landmarkToPaint:
        point1 = int(point[0])
        point2 = int(point[1])
        point_tuple = (point1, point2)
        name_landmark_str += ' %.1f %.1f' % (point[0], point[1])
        image_draw = cv2.circle(img_crop, point_tuple, point_size, point_color, thickness)
    if landmarkToPaint.shape[0] == 5:
        landmarks_image_path = r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(256,256)_move(0.250,0.000)_face_factor(0.500)_jpg\Landmark_painted_5_points"
    elif landmarkToPaint.shape[0] == 68:
        landmarks_image_path = r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(256,256)_move(0.250,0.000)_face_factor(0.500)_jpg\Landmark_painted_68_points"
    landmarks_image_path = os.path.join(landmarks_image_path, name)
    imwrite(landmarks_image_path, image_draw)

# ==============================================================================
# =                                opencv first                                =
# ==============================================================================

_DEAFAULT_JPG_QUALITY = 95
try:
    import cv2
    imread = cv2.imread
    imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEAFAULT_JPG_QUALITY])
    align_crop = cropper.align_crop_opencv
    print('Use OpenCV')
except:
    import skimage.io as io
    imread = io.imread
    imwrite = partial(io.imsave, quality=_DEAFAULT_JPG_QUALITY)
    align_crop = cropper.align_crop_skimage
    print('Importing OpenCv fails. Use scikit-image')


# ==============================================================================
# =                                     run                                    =
# ==============================================================================

# count landmarks
with open(args.landmark_file) as f:
    line = f.readline()
n_landmark = len(re.split('[ ]+', line)[1:]) // 2

# read data
img_names = np.genfromtxt(args.landmark_file, dtype=np.str, usecols=0)
landmarks = np.genfromtxt(args.landmark_file, dtype=np.float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1, n_landmark, 2)
standard_landmark = np.genfromtxt(args.standard_landmark_file, dtype=np.float).reshape(n_landmark, 2)
standard_landmark[:, 0] += args.move_w
standard_landmark[:, 1] += args.move_h

# data dir
save_dir = os.path.join(args.save_dir, 'align_size(%d,%d)_move(%.3f,%.3f)_face_factor(%.3f)_%s' % (args.crop_size_h, args.crop_size_w, args.move_h, args.move_w, args.face_factor, args.save_format))
data_dir = os.path.join(save_dir, 'data')
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

#load celeba 5 point landamrk
celeba_standard_landmark = np.loadtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\standard_landmark_celeba.txt", delimiter=',').reshape(-1, 5, 2)
celeba_landmark = np.genfromtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\list_landmarks_celeba.txt", dtype=np.float,usecols = range(1, 5 * 2 + 1), skip_header = 2).reshape(-1, 5, 2)


def work(i):  # a single work
    for _ in range(3):  # try three times
        try:
            img = imread(os.path.join(args.img_dir, img_names[i]))
            lm = landmarks[0]
            img_crop, tformed_landmarks, tformed_celeba_landmarks = align_crop(img,
                                                     landmarks[i],
                                                     standard_landmark,
                                                     celeba_standard_landmark,
                                                     celeba_landmark[i],
                                                     crop_size=(args.crop_size_h, args.crop_size_w),
                                                     face_factor=args.face_factor,
                                                     align_type=args.align_type,
                                                     order=args.order,
                                                     mode=args.mode)

            name = os.path.splitext(img_names[i])[0] + '.' + args.save_format
            path = os.path.join(data_dir, name)
            if not os.path.isdir(os.path.split(path)[0]):
                os.makedirs(os.path.split(path)[0])
            imwrite(path, img_crop)
            #tformed_celeba_landmarks.shape = -1


            # #draw face's landmark
            # point_size = 1
            # point_color = (0, 0, 255)  # BGR
            # thickness = 4  # 可以为 0 、4、8
            # name_landmark_str = name
            # for point in tformed_landmarks:
            #     point1 = int(point[0])
            #     point2 = int(point[1])
            #     point_tuple = (point1, point2)
            #     name_landmark_str += ' %.1f %.1f' % (point[0], point[1])
            #     image_draw = cv2.circle(img_crop, point_tuple, point_size, point_color, thickness)
            #
            # landmarks_image_path = r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(512,512)_move(0.250,0.000)_face_factor(0.500)_jpg\Landmark_pinated"
            # landmarks_image_path = os.path.join(landmarks_image_path, name)
            # imwrite(landmarks_image_path, image_draw)

             #store 68point with image name

            # %s %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f

            #draw the landmark before transformed
            if i <= 500:
                draw_face_landmark(name, tformed_celeba_landmarks, img_crop)
                draw_face_landmark(name, tformed_landmarks, img_crop)
            tformed_landmarks.shape = -1
            tformed_celeba_landmarks.shape = -1
            name_landmark_str = ('%s' + (' %.1f' * n_landmark * 2)) % ((name,) + tuple(tformed_landmarks))
            succeed = True
            break
        except Exception as e:
            succeed = False
            print(e.__traceback__)
    if succeed:
        return name_landmark_str, ' '.join([name] + [str(int(x)) for x in tformed_celeba_landmarks])
    else:
        print('%s fails!' % img_names[i])


if __name__ == '__main__':

    pool = Pool(args.n_worker)

    values = list(tqdm.tqdm(pool.imap(work, range(len(img_names))), total=len(img_names)))
    name_landmark_strs = []
    str2 = []
    for x in values:
        name_landmark_strs.append(x[0])
        str2.append(x[1])

    pool.close()
    pool.join()

    #np.savetxt("tformed_landmarks.txt", tformed_landmarks);
    landmarks_path = os.path.join(save_dir, 'tformed_landmark_68point.txt')
    with open(landmarks_path, 'w') as f:
        for name_landmark_str in name_landmark_strs:
            if name_landmark_str:
                f.write(name_landmark_str + '\n')

    #np.savetxt("tformed_landmarks.txt", tformed_landmarks);
    landmarks_path = os.path.join(save_dir, 'tformed_landmark_5point.txt')
    with open(landmarks_path, 'w') as f:
        for name_landmark_str in str2:
            if name_landmark_str:
                f.write(name_landmark_str + '\n')