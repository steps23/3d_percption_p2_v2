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
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from matplotlib.path import Path
import optuna

def get_calibrated_points(nusc, pointsensor_token, camera_token):
    """
    Carica il point cloud dal token del sensore LiDAR, lo trasforma nel sistema di coordinate della telecamera 
    e lo proietta sul piano 2D dell'immagine.
    """
    # Carica il point cloud
    pcl_path = nusc.get_sample_data_path(pointsensor_token)
    pc = LidarPointCloud.from_file(pcl_path)

    # Ottieni i record relativi al LiDAR e alla posa dell'ego
    lidar_sd_record = nusc.get('sample_data', pointsensor_token)
    cs_record_lidar = nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
    pose_record_lidar = nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])

    # Trasforma il point cloud dal frame del sensore LiDAR al frame del veicolo
    pc.rotate(Quaternion(cs_record_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record_lidar['translation']))

    # Trasforma il point cloud nel frame globale
    pc.rotate(Quaternion(pose_record_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(pose_record_lidar['translation']))

    # Ottieni i record della telecamera e della relativa posa
    cam_sd_record = nusc.get('sample_data', camera_token)
    cs_record_cam = nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])
    pose_record_cam = nusc.get('ego_pose', cam_sd_record['ego_pose_token'])

    # Trasforma il point cloud dal frame globale al frame dell'ego della telecamera
    pc.translate(-np.array(pose_record_cam['translation']))
    pc.rotate(Quaternion(pose_record_cam['rotation']).rotation_matrix.T)

    # Trasforma il point cloud dal frame dell'ego al frame della telecamera
    pc.translate(-np.array(cs_record_cam['translation']))
    pc.rotate(Quaternion(cs_record_cam['rotation']).rotation_matrix.T)

    # Proietta i punti sul piano immagine
    cam_intrinsic = np.array(cs_record_cam['camera_intrinsic'])
    points_3d_cam = pc.points[:3, :]
    points_2d = view_points(points_3d_cam, cam_intrinsic, normalize=True)

    return points_2d, points_3d_cam


def get_projected_boxes(nusc, camera_token):
    """
    Proietta le bounding boxes 3D sul piano dell'immagine.
    """
    cam_data = nusc.get('sample_data', camera_token)
    _, boxes, cam_intrinsic = nusc.get_sample_data(camera_token, box_vis_level=BoxVisibility.ALL)
    projected_boxes = []

    for box in boxes:
        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
        projected_boxes.append(corners_2d)

    return projected_boxes


def get_projected_boxes_with_labels(nusc, camera_token):
    """
    Proietta le bounding boxes 3D sul piano dell'immagine e restituisce le etichette originali.
    """
    cam_data = nusc.get('sample_data', camera_token)
    _, boxes, cam_intrinsic = nusc.get_sample_data(camera_token, box_vis_level=BoxVisibility.ALL)
    
    projected_boxes = []
    box_labels = []
    for box in boxes:
        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
        projected_boxes.append(corners_2d)
        box_labels.append(box.name)
    
    return projected_boxes, box_labels

def get_box_labels_from_camera(nusc, camera_token):
    """
    Estrae le etichette e le bounding boxes 2D degli oggetti per una data immagine della telecamera.
    """
    cam_data = nusc.get('sample_data', camera_token)
    my_sample = nusc.get('sample', cam_data['sample_token'])
    calibrated_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
    
    box_labels = []
    boxes_2d = []
    valid_tokens = []
    
    for ann_token in my_sample['anns']:
        box_3d = nusc.get('sample_annotation', ann_token)
        category_name = box_3d['category_name']
        box = nusc.get_box(ann_token)
        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
        if np.all(corners_2d[0, :] > 0) and np.all(corners_2d[0, :] < cam_data['width']) and \
           np.all(corners_2d[1, :] > 0) and np.all(corners_2d[1, :] < cam_data['height']):
            boxes_2d.append(corners_2d)
            box_labels.append(category_name)
            valid_tokens.append(ann_token)
    
    return box_labels, boxes_2d, valid_tokens


# CLUSTER ----------------- 

def cluster_lidar_points_in_boxes(points_2d, boxes_2d):
    """
    Per ogni box 2D (corners di shape (2,8)), restituisce 
    la lista degli indici di points_2d[:,i] che cadono dentro.

    Args:
        points_2d: np.ndarray di shape (3, N) o (2, N), con coordinate x, y (ed eventualmente z).
        boxes_2d:  list di np.ndarray, ciascuno di shape (2, 8) con i vertici della box proiettata.

    Returns:
        clusters: list di lunghezza len(boxes_2d), dove clusters[i] è la lista degli indici
                  di points_2d che ricadono nella i-esima box.
    """
    clusters = []
    # Prendi solo le coordinate x,y e trasponi in (N,2)
    pts = points_2d[:2, :].T  

    for box in boxes_2d:
        # box.T ha forma (8,2): i vertici nel formato richiesto da Path
        poly = Path(box.T)
        # mask[i] = True se pts[i] è dentro la polygon
        mask = poly.contains_points(pts)
        clusters.append(np.nonzero(mask)[0].tolist())

    return clusters      


#------------


def create_kf_2d(dt=0.5):
    """
    Crea un filtro di Kalman per un sistema in 2D con stato [x, y, vx, vy].
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 10.
    kf.R *= 0.01
    kf.Q = np.eye(4) * 0.01
    return kf


class ObjectTracker:
    """
    Classe per tracciare un oggetto (bounding box) utilizzando un filtro di Kalman
    con chi-square gating e filtro su incertezza.
    """
    def __init__(self, tracker_id, initial_measurement, dt=0.5):
        self.id = tracker_id
        self.kf = create_kf_2d(dt=dt)
        self.kf.x = np.array([
            initial_measurement[0],
            initial_measurement[1],
            0., 0.
        ]).reshape(4,1)
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.history = []

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.kf.x.copy())
        return self.kf.x
        
    def update(self, measurement):
        z = np.array(measurement).reshape(2,1)
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1


    def get_state(self):                          
        """
        Restituisce lo stato corrente [x, y, vx, vy] come vettore 1D.
        """
        return self.kf.x.flatten()



def associate_trackers_to_measurements(trackers, measurements, max_distance=50.0):
    """
    Associa i tracker alle misurazioni tramite distanza euclidea e l'algoritmo ungherese.
    Ritorna: matches, unmatched_measurements, unmatched_trackers.
    """
    if len(trackers) == 0 or len(measurements) == 0:
        return [], list(range(len(measurements))), list(range(len(trackers)))
    
    cost_matrix = np.zeros((len(trackers), len(measurements)), dtype=np.float32)
    for i, tracker in enumerate(trackers):
        pred_state = tracker.get_state()
        for j, meas in enumerate(measurements):
            cost_matrix[i, j] = np.linalg.norm(pred_state[:2] - meas)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_trackers = list(range(len(trackers)))
    unmatched_measurements = list(range(len(measurements)))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > max_distance:
            continue
        matches.append((r, c))
        if r in unmatched_trackers:
            unmatched_trackers.remove(r)
        if c in unmatched_measurements:
            unmatched_measurements.remove(c)
    
    return matches, unmatched_measurements, unmatched_trackers


def evaluate_tracking_rmse(scene_trackers_results, all_scenes_associations, scene_name):
    """
    Valuta quantitativamente il tracking per una scena confrontando i centri predetti dai tracker
    con i centri delle bounding boxes (misurazioni) ottenute dalle associazioni.
    
    Per ciascun frame:
      - Estrae i centri predetti dai tracker (prima e seconda componente dello stato).
      - Calcola il centro di ciascuna bounding box (usando np.mean sulle coordinate 2D dei vertici).
      - Utilizza l'algoritmo di assegnazione lineare per abbinare in modo ottimale i centri predetti
        alle misurazioni.
      - Calcola l'errore (distanza euclidea) per ciascun abbinamento e il RMSE per il frame.
    
    Parametri:
        scene_trackers_results: Dizionario dei risultati di tracking per ogni scena.
            Ogni frame è un dizionario contenente:
                - 'sample_token': token del frame.
                - 'tracker_states': lista degli stati dei tracker (vettori [x, y, vx, vy]).
        all_scenes_associations: Dizionario dei dati di associazione per ogni scena.
            Ogni frame è un dizionario contenente:
                - 'sample_token': token del frame.
                - 'boxes_2d': lista delle bounding boxes 2D (ciascuna con i 2D dei 8 vertici).
        scene_name: Chiave della scena da valutare (deve essere presente in entrambi i dizionari).
    
    Restituisce:
        overall_rmse: Errore quadratico medio complessivo calcolato su tutti i frame.
        per_frame_rmse: Lista degli RMSE calcolati per ogni frame.
    """

    rmses = []
    frame_results = scene_trackers_results[scene_name]
    frame_assocs = all_scenes_associations[scene_name]
    
    # Itera su ogni frame, abbinando per sample_token
    for tracker_frame in frame_results:
        sample_token = tracker_frame['sample_token']
        # Trova il frame associato con lo stesso sample_token
        assoc_frame = next((f for f in frame_assocs if f['sample_token'] == sample_token), None)
        if assoc_frame is None:
            continue
        
        # Estrai le predizioni dei tracker: utilizza solo [x, y]
        predictions = np.array([state[:2] for state in tracker_frame['tracker_states']])
        
        # Calcola le misurazioni: centro di ogni bounding box (media dei vertici)
        boxes = assoc_frame['boxes_2d']
        if len(boxes) == 0:
            continue
        measurements = np.array([np.mean(box, axis=1) for box in boxes])
        
        # Se non ci sono predizioni o misurazioni, passa al frame successivo
        if predictions.size == 0 or measurements.size == 0:
            continue
        
        # Costruisci la matrice dei costi (distanza euclidea tra ogni predizione e ogni misurazione)
        cost_matrix = np.linalg.norm(predictions[:, np.newaxis, :] - measurements[np.newaxis, :, :], axis=2)
        
        # Assegna in modo ottimale i tracker alle misurazioni
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Calcola l'errore per ciascun abbinamento
        errors = []
        for r, c in zip(row_ind, col_ind):
            errors.append(cost_matrix[r, c])
        if errors:
            frame_rmse = np.sqrt(np.mean(np.square(errors)))
            rmses.append(frame_rmse)
    
    if rmses:
        overall_rmse = np.mean(rmses)
    else:
        overall_rmse = None
    
    return overall_rmse, rmses

def create_kf_2d_tuned(dt, q_var, r_var, p_var):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P = np.eye(4) * p_var
    kf.R = np.eye(2) * r_var
    kf.Q = np.eye(4) * q_var
    return kf

# Sovrascrivete la funzione ObjectTracker per usare il vostro create_kf_2d_tuned
class ObjectTrackerTuned(ObjectTracker):
    def __init__(self, tracker_id, initial_measurement, dt, q_var, r_var, p_var):
        self.id = tracker_id
        self.kf = create_kf_2d_tuned(dt, q_var, r_var, p_var)
        self.kf.x = np.array([initial_measurement[0],
                              initial_measurement[1],
                              0., 0.]).reshape(4,1)
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.history = []

# Funzione che esegue il tracking su tutte le scene e restituisce l’RMSE
def run_tracking_and_evaluate(nusc, all_scenes_associations, 
                              dt, q_var, r_var, p_var,
                              max_distance, N_min, N_ref_dyn):
    """
    Esegue il tracking su tutte le scene e restituisce l’RMSE medio.
    """
    scene_trackers_results = {}
    object_id_counter = 0

    for scene_name, frames in all_scenes_associations.items():
        trackers = []
        frame_results = []
        for frame in frames:
            pts_2d = frame['lidar_points'][:2, :]
            boxes_2d = frame['boxes_2d']
            clusters = frame['clusters']
            measurements = []
            for box_idx, box in enumerate(boxes_2d):
                z_cam = np.mean(box, axis=1)
                pts_idx = clusters[box_idx]
                if len(pts_idx) >= N_min:
                    cluster = pts_2d[:, pts_idx]
                    z_lid = np.mean(cluster, axis=1)
                    w = min(len(pts_idx) / N_ref_dyn, 1.0)
                    z = w * z_lid + (1 - w) * z_cam
                else:
                    z = z_cam
                measurements.append(z)
            measurements = np.array(measurements)

            for tr in trackers:
                tr.predict()
            matches, unmatched_meas, _ = associate_trackers_to_measurements(
                trackers, measurements, max_distance=max_distance)
            for trk_idx, meas_idx in matches:
                trackers[trk_idx].update(measurements[meas_idx])
            trackers = [t for t in trackers if t.time_since_update <= 3]
            for meas_idx in unmatched_meas:
                new_trk = ObjectTrackerTuned(object_id_counter, measurements[meas_idx], dt, q_var, r_var, p_var)
                object_id_counter += 1
                trackers.append(new_trk)
            frame_results.append({
                'sample_token': frame['sample_token'],
                'tracker_states': [t.get_state() for t in trackers]
            })
        scene_trackers_results[scene_name] = frame_results

    rmses = []
    for scene_id in scene_trackers_results:
        overall_rmse, _ = evaluate_tracking_rmse(
            scene_trackers_results, all_scenes_associations, scene_id)
        if overall_rmse is not None:
            rmses.append(overall_rmse)
    return np.mean(rmses) if rmses else np.nan

def build_all_scenes_associations(nusc, camera_name='CAM_FRONT', lidar_name='LIDAR_TOP'):
    all_scenes_associations = {}
    for scene_idx, scene in enumerate(nusc.scene):
        scene_associations = []
        sample_token = scene['first_sample_token']
        while sample_token:
            sample = nusc.get('sample', sample_token)
            cam_token = sample['data'].get(camera_name)
            lidar_token = sample['data'].get(lidar_name)
            if cam_token and lidar_token:
                points_2d, _ = get_calibrated_points(nusc, lidar_token, cam_token)
                boxes_2d, box_labels = get_projected_boxes_with_labels(nusc, cam_token)
                clusters = cluster_lidar_points_in_boxes(points_2d, boxes_2d)
                scene_associations.append({
                    'sample_token': sample_token,
                    'camera_token': cam_token,
                    'lidar_points': points_2d,
                    'clusters': clusters,
                    'num_points': points_2d.shape[1],
                    'num_boxes': len(boxes_2d),
                    'box_labels': box_labels,
                    'boxes_2d': boxes_2d
                })
            sample_token = sample['next']
        all_scenes_associations[scene_idx] = scene_associations
    return all_scenes_associations

def objective(trial):
    # Sample hyperparameters from specified lists
    dt = trial.suggest_categorical('dt', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    q_var = trial.suggest_categorical('q_var', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    r_var = trial.suggest_categorical('r_var', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    p_var = trial.suggest_categorical('p_var', [1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    max_distance = trial.suggest_categorical('max_distance', [15.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    N_min = trial.suggest_categorical('N_min', [1, 4, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80])
    percentile = trial.suggest_categorical('percentile', [50, 60, 70, 75, 80, 85, 90, 95])

    # Load data (ensure heavy preprocessing is done outside objective for speed)
    nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes', verbose=False)
    associations = build_all_scenes_associations(nusc)

    # Compute dynamic reference N_ref_dyn
    all_sizes = [len(c) for frames in associations.values() for f in frames for c in f['clusters']]
    N_ref_dyn = int(np.percentile(all_sizes, percentile))

    # Evaluate RMSE
    mean_rmse = run_tracking_and_evaluate(
        nusc, associations,
        dt=dt, q_var=q_var, r_var=r_var, p_var=p_var,
        max_distance=max_distance,
        N_min=N_min,
        N_ref_dyn=N_ref_dyn
    )
    return mean_rmse

def main():
    # Create study with TPE sampler and median pruner
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

    # Optimize: 100 trials or 1 hour timeout
    study.optimize(objective, n_trials=100, timeout=3600)

    # Output best results
    print(f"Best RMSE: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save trials to CSV
    df_trials = study.trials_dataframe()
    df_trials.to_csv('optuna_study_results.csv', index=False)

if __name__ == '__main__':
    main()
