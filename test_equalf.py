import cv2
import h5py
import os
import numpy as np
import argparse
import yaml
import math
import time
import random
import statistics
from database import load_h5
from joblib import Parallel, delayed
from tqdm import tqdm
from errors import calc_mAA_pose, essential_pose_error_focal
import pygcransac

def get_LAF(kps, sc, ori):
    '''
    Converts OpenCV keypoints into transformation matrix
    and pyramid index to extract from for the patch extraction
    '''
    num = len(kps)
    out = np.zeros((num, 2, 2)).astype(np.float64)
    for i, kp in enumerate(kps):
        s = 12.0 * sc[i]
        a = ori[i]
        cos = math.cos(a) #  
        sin = math.sin(a) # 
        out[i, 0, 0] = s * cos
        out[i, 0, 1] = -s * sin
        out[i, 1, 0] = s * sin
        out[i, 1, 1] = s * cos
    return out

def run_gcransac(pair, matches, relative_pose, image_size1, image_size2, K1, K2, args):
    # Initialize the errors
    rotation_error = 1e10
    translation_error = 1e10
    numofmatches = 1.1
    inlier_number = 0.1
    focal_error = 1e10
    scalar = np.max(matches[:, :4])
    focal1 = K1[0,0]
    focal2 = K2[0,0]
    rf = focal2 / focal1 # To make image1 and image2 have the same focal length
    
    # Run point-based solver
    if args.solver == 0 or args.solver == 6:
        # The point correspondences
        matches[:,0] = (matches[:,0] - K1 [0,2])/scalar * rf
        matches[:,1] = (matches[:,1] - K1 [1,2])/scalar * rf
        matches[:,2] = (matches[:,2] - K2 [0,2])/scalar
        matches[:,3] = (matches[:,3] - K2 [1,2])/scalar
        
        correspondences = matches[:, :4].astype(np.float32)  
    # Run SIFT-based solver
    elif args.solver == 1:
        # The point correspondences
        matches[:,0] = (matches[:,0] - K1 [0,2])/scalar * rf
        matches[:,1] = (matches[:,1] - K1 [1,2])/scalar * rf
        matches[:,2] = (matches[:,2] - K2 [0,2])/scalar
        matches[:,3] = (matches[:,3] - K2 [1,2])/scalar
        points = matches[:, :4]

        # SIFT angles in the source image
        angle1 = matches[:, 4]  
        # SIFT angles in the destinstion image
        angle2 = matches[:, 5]  
        # SIFT scale in the source image
        scale1 = matches[:, 6] /scalar
        # SIFT scale in the destination image
        scale2 = matches[:, 7] /scalar

            

        # The point correspondences
        correspondences = np.concatenate((points, np.expand_dims(scale1, axis=1), np.expand_dims(scale2, axis=1), np.expand_dims(angle1, axis=1), np.expand_dims(angle2, axis=1)), axis=1)
    # Run affine-based solver refined
    elif args.solver == 2:

        matches[:,0] = (matches[:,0] - K1 [0,2])/scalar * rf
        matches[:,1] = (matches[:,1] - K1 [1,2])/scalar * rf
        matches[:,2] = (matches[:,2] - K2 [0,2])/scalar
        matches[:,3] = (matches[:,3] - K2 [1,2])/scalar
        points = matches[:, :4]
        # SIFT angles in the source image
        angle1 = matches[:, 4]  
        # SIFT angles in the destinstion image
        angle2 = matches[:, 5]  
        # SIFT scale in the source image
        scale1 = matches[:, 6]/scalar
        # SIFT scale in the destination image
        scale2 = matches[:, 7]/scalar

        laf1 = get_LAF(points[:, :2], scale1, angle1)
        laf2 = get_LAF(points[:, 2:4], scale2, angle2)

        # Calculating the affine transformation from the source to the destination image
        affines = [np.matmul(a2, np.linalg.inv(a1)) for a1, a2 in zip(laf1, laf2)]
        affines = np.array(affines).reshape(-1, 4)

        # Constructing the correspondences
        correspondences = np.concatenate((points, affines), axis=1)
    
    # Run monodepth-based solver
    elif args.solver == 5 or args.solver == 7 or args.solver == 8:
        # The point correspondences
        matches[:,0] = (matches[:,0] - K1 [0,2])/scalar * rf
        matches[:,1] = (matches[:,1] - K1 [1,2])/scalar * rf
        matches[:,2] = (matches[:,2] - K2 [0,2])/scalar
        matches[:,3] = (matches[:,3] - K2 [1,2])/scalar
        points = matches[:, :4]

        # SIFT angles in the source image
        angle1 = matches[:, 4]  
        # SIFT angles in the destinstion image
        angle2 = matches[:, 5]  
        # depths
        if args.depth == 0: # SIFT
            # SIFT scale in the source image
            scale1 = matches[:, 6] 
            # SIFT scale in the destination image
            scale2 = matches[:, 7] 
        elif args.depth == 1: # real depth
            scale1 = matches[:, 8]  
            scale2 = matches[:, 9]  
        elif args.depth == 2: # midas
            scale1 = matches[:, 10] 
            scale2 = matches[:, 11] 
        elif args.depth == 3: # dpt
            scale1 = matches[:, 12] 
            scale2 = matches[:, 13] 
        elif args.depth == 4: # zoedepth
            scale1 = matches[:, 14] 
            scale2 = matches[:, 15] 
        elif args.depth == 5: # depth anyV1B
            scale1 = matches[:, 16] 
            scale2 = matches[:, 17] 
        elif args.depth == 6: # depth anyV2B
            scale1 = matches[:, 18] 
            scale2 = matches[:, 19] 
        elif args.depth == 7: # depth-pro
            scale1 = matches[:, 20] 
            scale2 = matches[:, 21] 
        elif args.depth == 8: # metric3d V2
            scale1 = matches[:, 22] 
            scale2 = matches[:, 23] 
        elif args.depth == 9: # marigold e2e
            scale1 = matches[:, 24] 
            scale2 = matches[:, 25] 
        elif args.depth == 10: # moge
            scale1 = matches[:, 26] 
            scale2 = matches[:, 27] 
        elif args.depth == 11: # marigold
            scale1 = matches[:, 28] 
            scale2 = matches[:, 29] 
        elif args.depth == 12: # unidepth
            scale1 = matches[:, 30] 
            scale2 = matches[:, 31] 

        # The point correspondences
        correspondences = np.concatenate((points, np.expand_dims(scale1, axis=1), np.expand_dims(scale2, axis=1), np.expand_dims(angle1, axis=1), np.expand_dims(angle2, axis=1)), axis=1)
    
    # Inlier probabilities
    probabilities = []

    # Return if there are fewer than 8 correspondences
    if correspondences.shape[0] < 8:
        return 0, rotation_error, translation_error, inlier_number, numofmatches, focal_error

    # Run the fundamental matrix estimation implemented in gcransac
    tic = time.perf_counter()
    E_est, inliers = pygcransac.findFundamentalMatrix(
        np.ascontiguousarray(correspondences), 
        int(image_size1[0][0]), int(image_size1[1][0]), 
        int(image_size2[0][0]), int(image_size2[1][0]),
        use_sprt = False,
        probabilities = probabilities,
        spatial_coherence_weight = args.spatial_coherence_weight,
        neighborhood_size = args.neighborhood_size,
        conf = args.confidence,
        sampler = args.sampler,
        min_iters = args.minimum_iterations,
        max_iters = args.maximum_iterations,
        neighborhood = 0,
        solver = args.solver,
        lo_number = args.lo_number,
        threshold = args.inlier_threshold/scalar)
    
    toc = time.perf_counter()
    runtime = toc - tic
    
    # Count the inliers
    inlier_number = inliers.sum() 
    # Calculate the error if enough inliers are found
    if inlier_number >= 4:  
        # The original matches without SNN filtering
        numofmatches = matches.shape[0]
        # Calculate the pose error of the estimated fundamental matrix given the ground truth relative pose
        rotation_error, translation_error, focal_error = essential_pose_error_focal(E_est, scalar, relative_pose, K1, K2)


    return runtime, rotation_error, translation_error, inlier_number, numofmatches, focal_error

def estimate_essential(pairs, data, args):
    assert args.inlier_threshold > 0
    assert args.maximum_iterations > 0
    
    # Initialize the arrays where the results will be stored
    times = {}
    rotation_errors = {}
    translation_errors = {}
    inlier_numbers = {}
    nmatches = {}
    ferror = {}

    # Run fundamental matrix estimation on all image pairs
    keys = range(len(pairs))
    # keys = range(1)
    results = Parallel(n_jobs=min(args.core_number, len(pairs)))(delayed(run_gcransac)(
        pairs[k], # Name of the current pair
        data[f'corr_{pairs[k]}'], # The SIFT correspondences
        data[f'pose_{pairs[k]}'], # The ground truth relative pose coming from the COLMAP reconstruction
        data[f"size_{ '_'.join(pairs[k].split('_')[0:3]) }"], # The size of the source image
        data[f"size_{ '_'.join(pairs[k].split('_')[3:6]) }"], # The size of the destination image
        data[f"K_{ '_'.join(pairs[k].split('_')[0:3]) }"], # The intrinsic matrix of the source image
        data[f"K_{ '_'.join(pairs[k].split('_')[3:6]) }"], # The intrinsic matrix of the destination image
        args) for k in tqdm(keys)) # Other parameters

    for i, k in enumerate(keys):
        times[k] = results[i][0]
        rotation_errors[k] = results[i][1]
        translation_errors[k] = results[i][2]
        inlier_numbers[k] = results[i][3]
        if (len(results[i])<5):
            nmatches[k] = 0
        else:
            nmatches[k] = results[i][4]
        ferror[k] = results[i][5]
    return times, rotation_errors, translation_errors, inlier_numbers, nmatches, ferror

def sampler_flag(name):
    if name == "Uniform":
        return 0
    return 0

def depth_flag(name):
    if name == "depth":
        return 1
    if name == "midas":
        return 2
    if name == "dpt":
        return 3
    if name == "zoedepth":
        return 4
    if name == "anyv1":
        return 5
    if name == "anyv2":
        return 6
    if name == "depthpro":
        return 7
    if name == "metricV2":
        return 8
    if name == "marie2e":
        return 9
    if name == "moge":
        return 10
    if name == "marigold":
        return 11
    if name == "unidepth":
        return 12
    return 13

def solver_flag(name):
    if name == "point":
        return 0
    if name == "sift":
        return 1
    if name == "affine":
        return 2
    if name == "3p3d":
        return 5
    if name == "6pt":
        return 6
    if name == "4ptG":
        return 7
    if name == "4ptE":
        return 8
    return 100

if __name__ == "__main__":
    # Passing the arguments 
    parser = argparse.ArgumentParser(description="Running")
    parser.add_argument('--path', type=str, help="The path to the dataset. ", default="/")
    parser.add_argument('--split', type=str, help='Choose a split: splg, siftnn', default='splg', choices=['siftlg','splg','spnn','siftnn'])
    parser.add_argument('--scene', type=str, help='Choose a scene.', default='british_museum', choices=["all","british_museum","florence_cathedral", "lincoln_memorial", "london_bridge","milan_cathedral","mount_rushmore","piazza_sanmarco","sagrada_familia","stpauls_cathedral"])
    parser.add_argument("--config_path", type=str, default='dataset_configuration.yaml')
    parser.add_argument("--do_final_iterated_least_squares", type=bool, default=False) 
    parser.add_argument("--use_inlier_limit", type=bool, default=False)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--inlier_threshold", type=float, default=1)  
    parser.add_argument("--minimum_iterations", type=int, default=100)
    parser.add_argument("--maximum_iterations", type=int, default=100)
    parser.add_argument("--sampler", type=str, help="Choose from: Uniform, PROSAC, PNAPSAC, Importance, ARSampler.", choices=["Uniform"], default="Uniform")
    parser.add_argument("--spatial_coherence_weight", type=float, default=0.1)
    parser.add_argument("--neighborhood_size", type=float, default=2)
    parser.add_argument("--lo_number", type=int, default=0)
    parser.add_argument("--solver", type=str, help="Choose from: point, sift, affine.", choices=["point", "sift", "affine", "3p3d", "4ptG", "4ptE", "6pt"], default="4ptG")
    parser.add_argument("--depth", type=str, help="Choose from: sift, mono, depth.", choices=["sift", "depth", "midas", "dpt", "zoedepth", "anyv1", "anyv2", "marie2e", "moge", "depthpro", "metricV2", "marigold", "unidepth"], default="depth")
    parser.add_argument("--core_number", type=int, default=12)
    
    args = parser.parse_args()
    
    split = args.split.upper()
    print(f"Running GC-RANSAC on the '{split}' split of data")
    args.sampler = sampler_flag(args.sampler)
    # Loading the configuration file
    with open(args.config_path, "r") as stream:
        try:
            configuration = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()               
    # Iterating through the scenes
    for scene in configuration[f'{split}_SCENES']:
        # Check if the method should run on a single scene
        if args.scene != "all" and scene['name'] != args.scene:
            continue
        
        print(100 * "-")
        print(f"Loading scene '{scene['name']}'")
        print(100 * "-")
        
        # Loading the dataset
        input_fname = os.path.join(args.path, args.split, scene['filename'])
        data = load_h5(input_fname)
        pairs = sorted([x.replace('corr_','') for x in data.keys() if x.startswith('corr_')])
        print(f"{len(pairs)} image pairs are loaded.")


        args.solver = solver_flag(args.solver)
        args.depth = depth_flag(args.depth)
        times, rotation_errors, translation_errors, inlier_numbers, nmatches, ferror = estimate_essential(pairs, data, args)
        
        # Calculating the pose error as the maximum of the rotation and translation errors
        maximum_pose_errors = {}
        for i in range(len(pairs)):
            maximum_pose_errors[i] = max(rotation_errors[i], translation_errors[i])

        # Calculating the mean Average Accuracy
        mAA_translation_errors = calc_mAA_pose(translation_errors)
        mAA_rotation = calc_mAA_pose(rotation_errors)
        mean_rot = np.mean(np.array(list(rotation_errors.values())))
        median_rot = np.median(np.array(list(rotation_errors.values())))
        mean_f = np.mean(np.array(list(ferror.values())))
        median_f = np.median(np.array(list(ferror.values())))
        mean_trans = np.mean(np.array(list(translation_errors.values())))
        median_trans = np.median(np.array(list(translation_errors.values())))

        mAA_translation_errors5 = calc_mAA_pose(translation_errors, ths=np.linspace(0.1, 5, 10))
        mAA_rotation5 = calc_mAA_pose(rotation_errors, ths=np.linspace(0.1, 5, 10))
        
        mAA_f_errors01 = calc_mAA_pose(ferror, ths=np.linspace(0.1, 0.1, 10))
        mAA_f_errors02 = calc_mAA_pose(ferror, ths=np.linspace(0.1, 0.2, 10))

        print(f"mAA rotation error@5 = {mAA_rotation5:0.4f}")
        print(f"mAA translation error@5 = {mAA_translation_errors5:0.4f}")
        print(f"mAA rotation error@10 = {mAA_rotation:0.4f}")
        print(f"mAA translation error@10 = {mAA_translation_errors:0.4f}")
        print(f"mAA f error@01 = {mAA_f_errors01:0.4f}")
        print(f"mAA f error@02 = {mAA_f_errors02:0.4f}")
        print(f"mean rotation error = {mean_rot:0.4f}")
        print(f"median rotation error = {median_rot:0.4f}")
        print(f"mean translation error = {mean_trans:0.4f}")
        print(f"median translation error = {median_trans:0.4f}")
        print(f"mean f error = {mean_f:0.4f}")
        print(f"median f error = {median_f:0.4f}")
        print(f"Avg. inlier number = {np.mean(list(inlier_numbers.values())):0.1f}")
        print(f"Avg. run-time = {np.mean(list(times.values())):0.6f} secs")