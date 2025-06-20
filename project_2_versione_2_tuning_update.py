import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from IPython.display import HTML
import cv2
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes import NuScenesExplorer
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from matplotlib.path import Path
import optuna

# ------------------ Calibrated Points via Explorer ------------------
def get_calibrated_points(explorer, lidar_token, camera_token):
    """
    Uses the NuScenesExplorer helper to map LiDAR points into camera image plane.
    Returns:
      points_2d: numpy array of shape (3, N) - homogeneous image coords
    """
    points_tuple = explorer.map_pointcloud_to_image(
        pointsensor_token=lidar_token,
        camera_token=camera_token
    )
    points_2d = points_tuple[0]  # (3, N)
    return points_2d

# ------------------ Box Projection & Labels ------------------
def get_projected_boxes_with_labels(nusc, camera_token):
    _, boxes, cam_intrinsic = nusc.get_sample_data(camera_token, box_vis_level=BoxVisibility.ALL)
    projected, labels = [], []
    for box in boxes:
        corners_2d = view_points(box.corners(), cam_intrinsic, normalize=True)[:2, :]
        projected.append(corners_2d)
        labels.append(box.name)
    return projected, labels

# ------------------ Clustering ------------------
def cluster_lidar_points_in_boxes(points_2d, boxes_2d):
    pts = points_2d[:2, :].T  # (N,2)
    clusters = []
    for box in boxes_2d:
        poly = Path(box.T)
        mask = poly.contains_points(pts)
        clusters.append(np.nonzero(mask)[0].tolist())
    return clusters

# ------------------ Kalman Filter & Tracker ------------------
def create_kf_2d_tuned(dt, q_var, r_var, p_var):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    kf.H = np.array([[1,0,0,0],[0,1,0,0]])
    kf.P = np.eye(4) * p_var
    kf.R = np.eye(2) * r_var
    kf.Q = np.eye(4) * q_var
    return kf

class ObjectTrackerTuned:
    def __init__(self, tracker_id, initial_measurement, dt, q_var, r_var, p_var):
        self.id = tracker_id
        self.kf = create_kf_2d_tuned(dt, q_var, r_var, p_var)
        self.kf.x = np.array([initial_measurement[0], initial_measurement[1], 0., 0.]).reshape(4,1)
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.history = []
    def predict(self):
        self.kf.predict()
        self.age += 1; self.time_since_update += 1
        self.history.append(self.kf.x.copy())
        return self.kf.x
    def update(self, measurement):
        z = np.array(measurement).reshape(2,1)
        self.kf.update(z)
        self.time_since_update = 0; self.hits += 1
    def get_state(self): return self.kf.x.flatten()

# ------------------ Association & Evaluation ------------------
def associate_trackers_to_measurements(trackers, measurements, max_distance=50.0):
    if not trackers or measurements.size==0:
        return [], list(range(len(measurements))), list(range(len(trackers)))
    cost = np.zeros((len(trackers), len(measurements)))
    for i, tr in enumerate(trackers):
        pred = tr.get_state()[:2]
        for j, meas in enumerate(measurements): cost[i,j] = np.linalg.norm(pred - meas)
    row, col = linear_sum_assignment(cost)
    matches, um, ut = [], list(range(len(measurements))), list(range(len(trackers)))
    for r,c in zip(row,col):
        if cost[r,c] <= max_distance:
            matches.append((r,c)); um.remove(c); ut.remove(r)
    return matches, um, ut


def evaluate_tracking_rmse(scene_results, associations, scene_id):
    errors=[]
    for fr_tr, fr_ass in zip(scene_results[scene_id], associations[scene_id]):
        preds = np.array([s[:2] for s in fr_tr['tracker_states']])
        if preds.size==0 or not fr_ass['boxes_2d']: continue
        measures = np.array([np.mean(b,axis=1) for b in fr_ass['boxes_2d']])
        C = np.linalg.norm(preds[:,None,:] - measures[None,:,:], axis=2)
        r,c = linear_sum_assignment(C)
        errs=[C[i,j] for i,j in zip(r,c)]
        errors.append(np.sqrt(np.mean(np.square(errs))))
    return np.mean(errors) if errors else np.nan

# ------------------ Data Prep ------------------
def build_all_scenes_associations(nusc, explorer, camera_name='CAM_FRONT', lidar_name='LIDAR_TOP'):
    all_assoc={}
    for idx, scene in enumerate(nusc.scene):
        seq=[]; token=scene['first_sample_token']
        while token:
            samp=nusc.get('sample', token)
            cam_token = samp['data'].get(camera_name)
            lid_token = samp['data'].get(lidar_name)
            if cam_token and lid_token:
                pts2d = get_calibrated_points(explorer, lid_token, cam_token)
                boxes, labels = get_projected_boxes_with_labels(nusc, cam_token)
                clusters = cluster_lidar_points_in_boxes(pts2d, boxes)
                seq.append({'sample_token':token,'camera_token':cam_token,'lidar_points':pts2d,'boxes_2d':boxes,'box_labels':labels,'clusters':clusters})
            token = samp['next']
        all_assoc[idx]=seq
    return all_assoc

# ------------------ Main & Optimization ------------------
def run_tracking_and_evaluate(nusc, explorer, associations,
                              dt, q_var, r_var, p_var,
                              max_dist, N_min, N_ref_dyn):
    scene_res={}; obj_id=0
    for sid, frames in associations.items():
        trackers=[]; fr_res=[]
        for fr in frames:
            pts=fr['lidar_points'][:2,:]
            meas=[]
            for i,box in enumerate(fr['boxes_2d']):
                cam_c = np.mean(box,axis=1)
                idxs = fr['clusters'][i]
                if len(idxs)>=N_min:
                    lid_c=np.mean(pts[:,idxs],axis=1)
                    w=min(len(idxs)/N_ref_dyn,1.0)
                    z=w*lid_c + (1-w)*cam_c
                else:
                    z=cam_c
                meas.append(z)
            M=np.array(meas)
            # predict
            for t in trackers: t.predict()
            # associate
            matches, um, _ = associate_trackers_to_measurements(trackers, M, max_dist)
            for ti,mi in matches: trackers[ti].update(M[mi])
            trackers=[t for t in trackers if t.time_since_update<=3]
            for mi in um:
                trackers.append(ObjectTrackerTuned(obj_id, M[mi], dt, q_var, r_var, p_var)); obj_id+=1
            fr_res.append({'sample_token':fr['sample_token'],'tracker_states':[t.get_state() for t in trackers]})
        scene_res[sid]=fr_res
    # eval across scenes
    rmses=[evaluate_tracking_rmse(scene_res, associations, sid) for sid in scene_res]
    return np.nanmean(rmses)


def objective(trial):
    # hyperparams
    dt = trial.suggest_categorical('dt',[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    q_var = trial.suggest_categorical('q_var',[1e-5,1e-4,1e-3,1e-2,1e-1])
    r_var = trial.suggest_categorical('r_var',[1e-5,1e-4,1e-3,1e-2,1e-1,1.0])
    p_var = trial.suggest_categorical('p_var',[1.0,5.0,10.0,20.0,40.0,60.0,80.0,100.0])
    max_dist = trial.suggest_categorical('max_distance',[15.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0])
    N_min = trial.suggest_categorical('N_min',[1,4,8,10,15,20,25,30,40,50,60,70,80])
    percentile = trial.suggest_categorical('percentile',[50,60,70,75,80,85,90,95])
    # data
    nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes', verbose=False)
    explorer = NuScenesExplorer(nusc)
    assoc = build_all_scenes_associations(nusc, explorer)
    all_sizes=[len(c) for f in assoc.values() for fr in f for c in fr['clusters']]
    N_ref_dyn = int(np.percentile(all_sizes, percentile))
    return run_tracking_and_evaluate(nusc, explorer, assoc,
                                     dt, q_var, r_var, p_var,
                                     max_dist, N_min, N_ref_dyn)


def main():
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100, timeout=3600)
    print(f"Best RMSE: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k,v in study.best_params.items(): print(f"  {k}: {v}")
    pd.DataFrame(study.trials_dataframe()).to_csv('optuna_study_results.csv', index=False)

if __name__=='__main__':
    main()
