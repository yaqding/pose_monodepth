from typing import Dict
import numpy as np
import cv2
import math

def calc_mAA(MAEs, ths = np.logspace(np.log2(1.0), np.log2(20), 10, base=2.0)):
    res = 0
    cur_results = []
    for k, MAE in MAEs.items():
        acc = []
        for th in ths:
            A = (MAE <= th).astype(np.float32).mean()
            acc.append(A)
        cur_results.append(np.array(acc).mean())
    res = np.array(cur_results).mean()
    return res

def calc_mAA_pose(MAEs, ths = np.linspace(1.0, 10, 10)):
    res = 0
    cur_results = []
    if isinstance(MAEs, dict):
        for k, MAE in MAEs.items():
            acc = []
            for th in ths:
                A = (MAE <= th).astype(np.float32).mean()
                acc.append(A)
            cur_results.append(np.array(acc).mean())
        res = np.array(cur_results).mean()
    else:
        acc = []
        for th in ths:
            A = (MAEs <= th).astype(np.float32).mean()
            acc.append(A)
        res = np.array(acc).mean()

    return res

def trapezoid_area(a, b, h):
    return (a + b) * h / 2

# From Mihai
def compute_auc(method, errors, thresholds=[0.001, 0.01, 0.1], verbose=False):
    n_images = len(errors)
    errors = np.sort(errors)
    if verbose:
        print('[%s]' % method)
    gt_acc = 0.001
    errors[errors <= gt_acc] = 0.000
    results = []
    for threshold in thresholds:
        previous_error = 0
        previous_score = 0
        mAA = 0

        # Initialization.
        previous_error = gt_acc
        previous_score = np.sum(errors <= gt_acc)
        mAA = gt_acc * previous_score
        for error in errors:
            # Skip initialization.
            if error <= gt_acc:
                continue
            if error > threshold:
                break
            score = previous_score + 1
            mAA += trapezoid_area(previous_score, score, error - previous_error)
            previous_error = error
            previous_score = score
        mAA += trapezoid_area(previous_score, previous_score, threshold - previous_error)
        mAA /= (threshold * n_images)
        if verbose:
            print('AUC @ %2.3f - %.2f percents' % (threshold, mAA * 100))
        results.append(mAA * 100)
    return results

def focal_from_F(F):
    f11 = F[0,0]
    f12 = F[0,1]
    f13 = F[0,2]
    f21 = F[1,0]
    f22 = F[1,1]
    f23 = F[1,2]
    f31 = F[2,0]
    f32 = F[2,1]
    f33 = F[2,2]

    a = f11*f12*f31*f33-f11*f13*f31*f32+f12*f12*f32*f33-f12*f13*f32*f32+f21*f22*f31*f33-f21*f23*f31*f32+f22*f22*f32*f33-f22*f23*f32*f32
    b = f12*f13*f33*f33-f13*f13*f32*f33+f22*f23*f33*f33-f23*f23*f32*f33
    if (-b/a > 0):
        focal = np.sqrt(-b/a)
    else:
        focal = 1
        
    return focal


def essential_pose_error_focalvar(F, scalar, pose, K1, K2):
    R = pose[:, 0:3]
    t = pose[:, 3]

    f1 = focal_from_F(F)
    f2 = focal_from_F(F.transpose())
    
    K3 = np.matrix([[f1, 0, 0], [0, f1, 0], [0, 0, 1]])
    K4 = np.matrix([[f2, 0, 0], [0, f2, 0], [0, 0, 1]])
    
    focal_error = np.sqrt((abs(f1*scalar - K1[0,0])/(K1[0,0])) * (abs(f2*scalar - K2[0,0])/(K2[0,0])))

    normalizedEssential = K4@F@K3 
    R1, R2, translations = cv2.decomposeEssentialMat(normalizedEssential)
    rotations = np.zeros((2,3,3))
    rotations[0] = R1
    rotations[1] = R2

    minRotationError = 1e10
    minTranslationError = 1e10
    minError = 1e10

    for i in range(len(rotations)):
        Rest = rotations[i]
        test = translations

        if np.isnan(Rest).any() or np.isnan(test).any():
            continue

        try:
            err_R, err_t = evaluate_R_t(R, t, Rest, test, q_gt=None)

            if err_R + err_t < minError:
                minError = err_R + err_t
                minRotationError = err_R
                minTranslationError = err_t
        except:
            print("Error!")
            continue
    
    return 180.0 / math.pi * minRotationError, 180.0 / math.pi * minTranslationError, focal_error

def essential_pose_error_focal(F, scalar, pose, K1, K2):
    R = pose[:, 0:3]
    t = pose[:, 3]

    focal = focal_from_F(F)
    K4 = np.matrix([[focal, 0, 0], [0, focal, 0], [0, 0, 1]])
    K3 = np.matrix([[focal, 0, 0], [0, focal, 0], [0, 0, 1]])
    
    focal_error = abs(focal - K2[0,0]/scalar)/(K2[0,0]/scalar)
    normalizedEssential = K4@F@K3 
    
    R1, R2, translations = cv2.decomposeEssentialMat(normalizedEssential)
    rotations = np.zeros((2,3,3))
    rotations[0] = R1
    rotations[1] = R2

    minRotationError = 1e10
    minTranslationError = 1e10
    minError = 1e10

    for i in range(len(rotations)):
        Rest = rotations[i]
        test = translations

        if np.isnan(Rest).any() or np.isnan(test).any():
            continue

        try:
            err_R, err_t = evaluate_R_t(R, t, Rest, test, q_gt=None)

            if err_R + err_t < minError:
                minError = err_R + err_t
                minRotationError = err_R
                minTranslationError = err_t
        except:
            print("Error!")
            continue
    
    return 180.0 / math.pi * minRotationError, 180.0 / math.pi * minTranslationError, focal_error


def essential_pose_error(Ess, pose):
    R = pose[:, 0:3]
    t = pose[:, 3]

    #retval, rotations, translations, normals = decomposeHomography(normalizedHomography)
    R1, R2, translations = cv2.decomposeEssentialMat(Ess)
    rotations = np.zeros((2,3,3))
    rotations[0] = R1
    rotations[1] = R2

    minRotationError = 1e10
    minTranslationError = 1e10
    minError = 1e10

    for i in range(len(rotations)):
        Rest = rotations[i]
        test = translations

        if np.isnan(Rest).any() or np.isnan(test).any():
            continue

        try:
            err_R, err_t = evaluate_R_t(R, t, Rest, test, q_gt=None)

            if err_R + err_t < minError:
                minError = err_R + err_t
                minRotationError = err_R
                minTranslationError = err_t
        except:
            print("Error!")
            continue
    
    return 180.0 / math.pi * minRotationError, 180.0 / math.pi * minTranslationError


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    # make det(R_gt) det(R) = 1
    sin_angle1 = np.linalg.norm(R_gt / np.cbrt(np.linalg.det(R_gt)) - R / np.cbrt(np.linalg.det(R)), 'fro') / (2.0 * math.sqrt(2.0))
    sin_angle = max(min(1.0, sin_angle1), -1.0)
    err_r = 2*math.asin(sin_angle)
    
    t = t / (np.linalg.norm(t))
    t_gt = t_gt / (np.linalg.norm(t_gt))
    err_t = min(2*math.asin(np.linalg.norm(t - t_gt)*0.5), 2*math.asin(np.linalg.norm(t + t_gt)*0.5))
        

    return err_r, err_t
