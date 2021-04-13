import numpy as np
from numpy import linalg as LA

def align_crop_opencv(img,
                      src_landmarks,
                      standard_landmarks,
                      celeba_standard_landmark,
                      src_celeba_landmark,
                      crop_size=512,
                      face_factor=0.7,
                      align_type='similarity',
                      order=3,
                      mode='edge'):
    """Align and crop a face image by landmarks.

    Arguments:
        img                : Face image to be aligned and cropped.
        src_landmarks      : [[x_1, y_1], ..., [x_n, y_n]].
        standard_landmarks : Standard shape, should be normalized.
        crop_size          : Output image size, should be 1. int for (crop_size, crop_size)
                             or 2. (int, int) for (crop_size_h, crop_size_w).
        face_factor        : The factor of face area relative to the output image.
        align_type         : 'similarity' or 'affine'.
        order              : The order of interpolation. The order has to be in the range 0-5:
                                 - 0: INTER_NEAREST
                                 - 1: INTER_LINEAR
                                 - 2: INTER_AREA
                                 - 3: INTER_CUBIC
                                 - 4: INTER_LANCZOS4
                                 - 5: INTER_LANCZOS4
        mode               : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                             Points outside the boundaries of the input are filled according
                             to the given mode.
    """
    # set OpenCV
    import cv2

    inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
             3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

    # crop size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_size_h = crop_size[0]
        crop_size_w = crop_size[1]
    elif isinstance(crop_size, int):
        crop_size_h = crop_size_w = crop_size
    else:
        raise Exception('Invalid `crop_size`! `crop_size` should be 1. int for (crop_size, crop_size) or 2. (int, int) for (crop_size_h, crop_size_w)!')



    # estimate transform matrix
    trg_landmarks = standard_landmarks * max(crop_size_h, crop_size_w) * face_factor + np.array([crop_size_w // 2, crop_size_h // 2])

    if align_type == 'affine':
        tform = cv2.estimateAffine2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]
    else:
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]

    #calcaute the scale of tform
    m1 = np.mat('0;0;1')
    m2 = np.mat('1;0;1')
    p1 = tform.dot(m1)
    p2 = tform.dot(m2)
    scale = LA.norm(p2 - p1) # defualt is Frobenius norm

    # change the translations part of the transformation matrix for downwarding vertically
    tform[1][2] = tform[1][2] + 20*scale


    # warp image by given transform
    output_shape = (crop_size_h, crop_size_w)
    img_crop = cv2.warpAffine(img, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order], borderMode=border[mode])

    # #center crop
    # center_crop_size = 224
    # mid_x, mid_y = int(crop_size_w / 2), int(crop_size_h / 2)
    # mid_y = mid_y +16
    # cw2, ch2 = int(center_crop_size / 2), int(center_crop_size / 2)
    # img_crop = img_crop[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]




    # get transformed landmarks
    tformed_landmarks = cv2.transform(np.expand_dims(src_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]
    tformed_celeba_landmarks = cv2.transform(np.expand_dims(src_celeba_landmark, axis=0), cv2.invertAffineTransform(tform))[0]

    return img_crop, tformed_landmarks,tformed_celeba_landmarks,tform


def align_crop_skimage(img,
                       src_landmarks,
                       standard_landmarks,
                       crop_size=512,
                       face_factor=0.7,
                       align_type='similarity',
                       order=3,
                       mode='edge'):
    """Align and crop a face image by landmarks.

    Arguments:
        img                : Face image to be aligned and cropped.
        src_landmarks      : [[x_1, y_1], ..., [x_n, y_n]].
        standard_landmarks : Standard shape, should be normalized.
        crop_size          : Output image size, should be 1. int for (crop_size, crop_size)
                             or 2. (int, int) for (crop_size_h, crop_size_w).
        face_factor        : The factor of face area relative to the output image.
        align_type         : 'similarity' or 'affine'.
        order              : The order of interpolation. The order has to be in the range 0-5:
                                 - 0: INTER_NEAREST
                                 - 1: INTER_LINEAR
                                 - 2: INTER_AREA
                                 - 3: INTER_CUBIC
                                 - 4: INTER_LANCZOS4
                                 - 5: INTER_LANCZOS4
        mode               : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                             Points outside the boundaries of the input are filled according
                             to the given mode.
    """
    raise NotImplementedError("'align_crop_skimage' is not implemented!")

#https://en.wikipedia.org/wiki/Procrustes_analysis
def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    # U, S, Vt = np.linalg.svd(points1.T * points2)
    # R = (U * Vt).T

    # return np.vstack([np.hstack(((s2 / s1) * R,
    #                                    c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])
    return np.around(points1,5), np.around(points2,5)

landmarks = np.genfromtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\list_landmarks_celeba.txt", usecols=range(1, 5 * 2 + 1),skip_header = 2).reshape(-1, 5, 2)
x_celeba_lanamrk = landmarks[:,:,0]
y_celeba_lanamrk = landmarks[:,:,1]
transformation_from_points(x_celeba_lanamrk,y_celeba_lanamrk)

