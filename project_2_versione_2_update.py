# %% [markdown]
# IMPORTAZIONI

# %%
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from IPython.display import HTML
from IPython.display import display
import cv2
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes import NuScenesExplorer
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

from filterpy.kalman import KalmanFilter

from scipy.optimize import linear_sum_assignment

from matplotlib.path import Path


# %%
def get_calibrated_points(explorer, lidar_token, camera_token):
    points_tuple = explorer.map_pointcloud_to_image(
        pointsensor_token=lidar_token,
        camera_token=camera_token
    )
    points_2d = points_tuple[0]  # solo i punti (3, N)
    return points_2d



def visualize_calibrated_points(nusc, camera_token, points_2d):
    """
    Visualizza i punti LiDAR calibrati proiettati sulla vista della telecamera.
    """
    cam = nusc.get('sample_data', camera_token)
    img_path = os.path.join(nusc.dataroot, cam['filename'])
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Filtra i punti che rientrano nelle dimensioni dell'immagine
    width, height = img_np.shape[1], img_np.shape[0]
    valid_points = (points_2d[0, :] >= 0) & (points_2d[0, :] < width) & \
                   (points_2d[1, :] >= 0) & (points_2d[1, :] < height)
    points_2d_filtered = points_2d[:, valid_points]

    plt.figure(figsize=(10, 6))
    plt.imshow(img_np)
    plt.scatter(points_2d_filtered[0, :], points_2d_filtered[1, :], c='red', s=5, label='Punti LiDAR Calibrati')
    plt.xlabel("Coordinata X (pixel)")
    plt.ylabel("Coordinata Y (pixel)")
    plt.legend()
    plt.title("Punti LiDAR Calibrati sulla Vista della Camera")
    plt.show()


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


def draw_3d_box_2d(ax, corners_2d, color='b', linewidth=2):
    """
    Disegna in 2D il parallelepipedo corrispondente alla box 3D proiettata,
    connessa da 12 segmenti (8 vertici).

    corners_2d: np.array di shape (2, 8), dove
                corners_2d[0, :] = x-coord di ciascun vertice
                corners_2d[1, :] = y-coord di ciascun vertice
    """
    # Se l'array non ha 8 vertici, esci
    if corners_2d.shape[1] != 8:
        return

    # Definisci gli spigoli che connettono i vertici per formare il parallelepipedo
    # L'ordine dei vertici in box.corners() di NuScenes di solito è:
    #   0----1
    #   |    |
    #   3----2    (in basso)
    #   4----5
    #   |    |
    #   7----6    (in alto)
    # Verifica se l'ordine differisce e adatta di conseguenza.
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # base
        (4,5), (5,6), (6,7), (7,4),  # top
        (0,4), (1,5), (2,6), (3,7)   # verticali
    ]

    for start, end in edges:
        x_coords = [corners_2d[0, start], corners_2d[0, end]]
        y_coords = [corners_2d[1, start], corners_2d[1, end]]
        ax.plot(x_coords, y_coords, color=color, linewidth=linewidth)


def visualize_fused_positions_on_image(nusc, camera_token, fused_positions, boxes_2d, box_labels):
    """
    Visualizza le posizioni fuse (da filtro di Kalman) sull'immagine della telecamera,
    insieme alle bounding boxes con le etichette.
    """
    cam = nusc.get('sample_data', camera_token)
    img_path = os.path.join(nusc.dataroot, cam['filename'])
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_np)
    plt.axis('off')

    unique_labels = list(set(box_labels))
    num_labels = len(unique_labels)
    colors = plt.cm.get_cmap('tab20', num_labels)
    label_color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    for i, box in enumerate(boxes_2d):
        label = box_labels[i]
        color = label_color_map[label]
        plt.plot(box[0, [0, 1, 2, 3, 0]],
                 box[1, [0, 1, 2, 3, 0]],
                 color=color,
                 linewidth=2)

    for i, pos in enumerate(fused_positions):
        plt.plot(pos[0], pos[1], 'x', color='red', markersize=10, markeredgewidth=2)
        plt.text(pos[0], pos[1] - 10, f'Fused {i + 1}', color='red', fontsize=12, ha='center')

    legend_patches = [mpatches.Patch(color=label_color_map[label], label=label) for label in unique_labels]
    plt.legend(handles=legend_patches, loc='upper right')
    plt.title("Fused Positions from Kalman Filter on Original Image")
    plt.show()

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


def visualize_clusters(nusc, camera_token, points_2d, boxes_2d, clusters, box_labels=None):
    # carica e mostra immagine
    # disegna tutte le box
    for i, box in enumerate(boxes_2d):
        draw_3d_box_2d(ax, box, color='b', linewidth=2)
    # disegna i punti per cluster
    for box_idx, pts_idx in enumerate(clusters):
        color = 'r' if box_labels is None else label_color_map[box_labels[box_idx]]
        pts = points_2d[:, pts_idx]
        ax.scatter(pts[0], pts[1], c=[color], s=5)
        

def animate_scene_clusters(nusc, all_scenes_associations, scene_name, interval=100):
    """
    Restituisce un'animazione FuncAnimation che mostra, frame per frame,
    i cluster di punti LiDAR all'interno delle box 2D su ciascuna immagine.
    """
    frames = all_scenes_associations[scene_name]
    fig, ax = plt.subplots(figsize=(12,8))

    # pre-calcola il colormap per le etichette
    unique_labels = list({lab for f in frames for lab in f['box_labels']})
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    colmap = {lab: cmap(i) for i,lab in enumerate(unique_labels)}

    def update(frame_idx):
        ax.clear()
        frame = frames[frame_idx]

        # carica immagine
        cam = nusc.get('sample_data', frame['camera_token'])
        img = Image.open(os.path.join(nusc.dataroot, cam['filename'])).convert("RGB")
        ax.imshow(img); ax.axis('off')

        # disegna le box
        for b in frame['boxes_2d']:
            draw_3d_box_2d(ax, b, color='b', linewidth=2)

        # disegna i cluster
        pts2d  = frame['lidar_points'][:2, :]  # (2,N)
        for i, pts_idx in enumerate(frame['clusters']):
            if not pts_idx: 
                continue
            pts = pts2d[:, pts_idx]
            lab = frame['box_labels'][i]
            ax.scatter(pts[0], pts[1], s=5, c=[colmap[lab]], label=lab)

        ax.legend(loc='upper right')
        ax.set_title(f"Frame {frame_idx+1}/{len(frames)}")

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=interval,
        repeat=False
    )
    return ani


#------------


def create_kf_2d(dt=0.5):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]]) # transizione dello stato
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]]) # osservazione
    # incertezza iniziale sullo stato
    kf.P = np.eye(4) * 10.0
    # rumore di misura
    kf.R = np.eye(2) * 0.0001
    # rumore di processo
    kf.Q = np.eye(4) * 0.1
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

def animate_scene_tracking(nusc, scene_trackers_results, all_scenes_associations, scene_name, interval=100):
    """
    Crea e restituisce un'animazione per il tracking di una scena, utilizzando i dati contenuti in scene_trackers_results.
    Per ciascun frame viene visualizzata l'immagine della camera con i tracker (stati: [x, y, vx, vy]) tracciati sopra.
    
    Parametri:
        nusc: Istanza NuScenes.
        scene_trackers_results: Dizionario con i risultati di tracking per ogni scena.
            Per ogni frame (dizionario) sono attesi i seguenti campi:
                - 'sample_token': token del frame.
                - 'tracker_states': lista degli stati dei tracker, dove ogni stato è un vettore [x, y, vx, vy].
        all_scenes_associations: Dizionario con i dati di associazione per ogni scena, utilizzato per ottenere il token della camera.
            Per ogni frame sono attesi i campi:
                - 'sample_token': token del frame.
                - 'camera_token': token della camera.
        scene_name: Nome della scena da animare (chiave presente sia in scene_trackers_results che in all_scenes_associations).
        interval: Intervallo in millisecondi tra un frame e l'altro.
    
    Restituisce:
        ani: Oggetto animation.FuncAnimation, visualizzabile (ad es. con HTML(ani.to_jshtml()))
    """
    # Estrae i dati dei frame per il tracking della scena scelta
    tracker_frames = scene_trackers_results[scene_name]
    assoc_frames = all_scenes_associations[scene_name]
    
    # Crea la figura e l'asse per l'animazione
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def update(frame_idx):
        ax.clear()
        if frame_idx >= len(tracker_frames):
            return
        
        # Estrae i dati del frame corrente per il tracking
        frame_tracking = tracker_frames[frame_idx]
        sample_token = frame_tracking['sample_token']
        tracker_states = frame_tracking['tracker_states']
        
        # Trova il frame associato per ottenere il token della camera
        frame_assoc = next((f for f in assoc_frames if f['sample_token'] == sample_token), None)
        if frame_assoc is None:
            ax.set_title("Nessun frame_assoc trovato!")
            return
        camera_token = frame_assoc['camera_token']
        
        # Carica e visualizza l'immagine della camera
        cam_data = nusc.get('sample_data', camera_token)
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        if not os.path.exists(img_path):
            ax.set_title(f"Immagine non trovata: {img_path}")
            return
        
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        ax.imshow(img_np)
        ax.set_title(f"Frame {frame_idx} - Sample Token: {sample_token}")
        ax.axis('off')
        
        # Disegna i tracker: per ciascun tracker visualizza il punto e il suo ID
        for t_idx, state in enumerate(tracker_states):
            x, y, vx, vy = state
            ax.plot(x, y, 'ro', markersize=6)
            ax.text(x, y - 10, f"ID {t_idx}", color='red', fontsize=10)
    
    ani = animation.FuncAnimation(fig, update, frames=len(tracker_frames), interval=interval)
    return ani

def get_gt_centers_2d(nusc, camera_token):
    """
    Per un singolo frame (camera_token):
    - Carica tutte le box 3D annotate (Box) con box_vis_level=ALL
    - Prende il centro 3D di ciascuna box (box.center: [x,y,z])
    - Proietta in 2D con la intrinseca della camera
    Restituisce un array (M,2) di [x_pixel, y_pixel].
    """
    _, boxes, cam_intrinsic = nusc.get_sample_data(
        camera_token,
        box_vis_level=BoxVisibility.ALL
    )
    gt_centers = []
    for box in boxes:
        # centro 3D nella camera frame
        c3 = box.center.reshape(3,1)              # (3,1)
        # proiezione in pixel
        c2 = view_points(c3, cam_intrinsic, normalize=True)[:2,0]
        gt_centers.append(c2)
    return np.array(gt_centers)  # shape (M,2)

def evaluate_tracking_rmse(scene_trackers_results, all_scenes_associations, scene_name, nusc):
    """
    Valuta il tracking calcolando l'RMSE fra:
      - le predizioni dei tracker (in pixel, [x,y])
      - i centri delle bounding box annotate proiettati in 2D (gt_centers_2d)
    
    Parametri:
      scene_trackers_results: dict scena -> lista frame con 'tracker_states'
      all_scenes_associations: dict scena -> lista frame con 'sample_token' e 'camera_token'
      scene_name: chiave della scena da valutare
      nusc: istanza di NuScenes
    
    Restituisce:
      overall_rmse: RMSE complessivo su tutti i frame
      per_frame_rmse: lista di RMSE per ciascun frame
    """
    per_frame = []
    tracker_frames = scene_trackers_results[scene_name]
    assoc_frames   = all_scenes_associations[scene_name]

    for trk_frame in tracker_frames:
        token = trk_frame['sample_token']
        # trova frame_assoc corrispondente
        assoc = next(f for f in assoc_frames if f['sample_token']==token)
        cam_token = assoc['camera_token']
        
        # predizioni tracker: lista di [x,y,vx,vy] -> prendo solo x,y
        preds = np.array([s[:2] for s in trk_frame['tracker_states']])
        if preds.size==0:
            continue
        
        # GT: centri proiettati
        gt_centers = get_gt_centers_2d(nusc, cam_token)
        if gt_centers.size==0:
            continue
        
        # Calcola la norma (distanza euclidea)
        # ogni elemento cost[i, j] è la distanza euclidea tra il tracker i e il ground-truth j.
        cost = np.linalg.norm(preds[:,None,:] - gt_centers[None,:,:], axis=2) 
        row, col = linear_sum_assignment(cost) #quale predizione corrisponde a quale ground truth
        
        # calcola errori per gli accoppiamenti validi
        errs = [ cost[r,c] for r,c in zip(row,col) ]
        frame_rmse = np.sqrt(np.mean(np.square(errs))) if errs else None
        if frame_rmse is not None:
            per_frame.append(frame_rmse)

    overall_rmse = np.mean(per_frame) if per_frame else None
    return overall_rmse, per_frame



# %%
# Inizializza il dataset NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes', verbose=False)
explorer = NuScenesExplorer(nusc)
# Seleziona una scena e un sample di esempio
my_scene = nusc.scene[0]
my_sample = nusc.sample[10]

# Ottieni i token del LiDAR e della camera per il sample corrente
pointsensor_token = my_sample['data']['LIDAR_TOP']
camera_token = my_sample['data']['CAM_FRONT']

# Estrai i punti proiettati e le bounding boxes con etichette
projected_points = get_calibrated_points(explorer, pointsensor_token, camera_token)
boxes_2d, box_labels = get_projected_boxes_with_labels(nusc, camera_token)

# Visualizza i punti LiDAR calibrati
visualize_calibrated_points(nusc, camera_token, projected_points)

# %%
# Per ogni frame di una scena, salva i dati di associazione
all_scenes_associations = {}
for scene_idx, scene in enumerate(nusc.scene):
    print(f"Elaboro scene_idx={scene_idx}, scene_name={scene['name']}")
    scene_associations = []
    sample_token = scene['first_sample_token']
    while sample_token:
        sample = nusc.get('sample', sample_token)
        cam_token = sample['data'].get('CAM_FRONT', None)
        lidar_token = sample['data'].get('LIDAR_TOP', None)
        if cam_token is None or lidar_token is None:
            sample_token = sample['next']
            continue
        points_2d = get_calibrated_points(explorer, lidar_token, cam_token)
        boxes_2d, box_labels = get_projected_boxes_with_labels(nusc, cam_token)
        clusters = cluster_lidar_points_in_boxes(points_2d, boxes_2d)
        frame_assoc = {
            'sample_token': sample_token,
            'camera_token': cam_token,
            'lidar_points': points_2d,  
            'clusters': clusters,
            'num_points': points_2d.shape[1],
            'num_boxes': len(boxes_2d),
            'box_labels': box_labels,
            'boxes_2d': boxes_2d
        }
        scene_associations.append(frame_assoc)
        sample_token = sample['next']
    all_scenes_associations[scene_idx] = scene_associations

print("Elaborazione associations completata!")
print("Scene elaborate:", list(all_scenes_associations.keys()))

# %%
scene_name = 4  # Sostituisci con un nome presente in all_scenes_associations
ani = animate_scene_clusters(nusc, all_scenes_associations, scene_name, interval=100)
n_frames = len(all_scenes_associations[scene_name])
mpl.rcParams['animation.embed_limit'] = n_frames
HTML(ani.to_jshtml(embed_frames=n_frames))

# %%
# SENSOR DATA FUSION CON FILTRO DI KALMAN (LiDAR+Camera measurement-level fusion pesata)
scene_trackers_results = {}
object_id_counter = 0
dt = 0.1
max_distance = 15.0
N_min = 70

all_sizes = [len(c) for frames in all_scenes_associations.values() for f in frames for c in f['clusters']]
N_ref_dyn = int(np.percentile(all_sizes, 95)) 
#print("--------------")
#print(N_ref_dyn)
#print("--------------")

for scene_name, frames in all_scenes_associations.items():
    print("Tracking per scena:", scene_name)
    trackers = []
    frame_results = []

    for frame in frames:
        # estraiamo i soli x,y dai punti LiDAR (ignoro la terza riga di view_points)
        raw_pts      = frame['lidar_points']    # shape (3, N)
        pts_2d       = raw_pts[:2, :]           # shape (2, N)
        boxes_2d     = frame['boxes_2d']
        clusters = frame['clusters']

        
        # --- Costruzione delle misure fuse LiDAR+camera (fusione pesata) ---
        measurements = []
        for box_idx, box in enumerate(boxes_2d):
            # centro della box (camera)
            z_cam = np.mean(box, axis=1)  # (2,)

            # punti LiDAR associati
            pts_idx = clusters[box_idx] 
            if len(pts_idx) >= N_min: # se ci almeno N_min punti nel cluster li prendo in considerazione
                #print(len(pts_idx))
                cluster = pts_2d[:, pts_idx]        # (2, K)
                z_lid   = np.mean(cluster, axis=1)  # (2,)
                # peso w in [0,1] proporzionale al numero di punti
                w = min(len(pts_idx) / N_ref_dyn, 1.0)
                # misura fusa
                z = w * z_lid + (1 - w) * z_cam
            else:
                # nessun punto LiDAR → uso solo camera
                z = z_cam

            measurements.append(z)

        measurements = np.array(measurements)  # shape (M, 2)

        # --- Prediction step per tutti i tracker attivi ---
        for tracker in trackers:
            tracker.predict()

        # --- Data association tracker ↔ misure ---
        matches, unmatched_meas, unmatched_trk = associate_trackers_to_measurements(trackers, measurements, max_distance=max_distance)

        # --- Update step ---
        for trk_idx, meas_idx in matches:
            trackers[trk_idx].update(measurements[meas_idx])

        # --- Rimuovi tracker (>3 frame senza update) ---
        trackers = [t for t in trackers if t.time_since_update <= 3]

        # --- Crea nuovi tracker per misure non abbinate ---
        for meas_idx in unmatched_meas:
            new_trk = ObjectTracker(object_id_counter, measurements[meas_idx], dt=dt)
            object_id_counter += 1
            trackers.append(new_trk)

        # --- Salva gli stati correnti per questo frame ---
        current_states = [t.get_state() for t in trackers]
        frame_results.append({
            'sample_token':   frame['sample_token'],
            'tracker_states': current_states
        })

    scene_trackers_results[scene_name] = frame_results

print("Tracking completato!")


# %%
# Esempio di animazione per il tracking (scene 0)
scene_name=4
ani_tracking = animate_scene_tracking(nusc, scene_trackers_results, all_scenes_associations, scene_name, interval=100)
n_frames = len(all_scenes_associations[scene_name])
mpl.rcParams['animation.embed_limit'] = n_frames
HTML(ani.to_jshtml(embed_frames=n_frames))
HTML(ani_tracking.to_jshtml())

# %%
#print(evaluate_tracking_rmse(scene_trackers_results, all_scenes_associations, scene_name))

# %%
"""overall_rmse, per_frame_rmse = evaluate_tracking_rmse(scene_trackers_results, all_scenes_associations, scene_name=4)

print("Risultati di valutazione del tracking:")
print(f"RMSE complessivo: {overall_rmse:.3f}\n")
print("RMSE per ciascun frame:")

for i, rmse in enumerate(per_frame_rmse):
    print(f"Frame {i+1:03d}: {rmse:.3f}")"""


# %%
overall_rmse, per_frame_rmse = evaluate_tracking_rmse(
    scene_trackers_results,
    all_scenes_associations,
    scene_name=2,
    nusc=nusc
)
print("RMSE complessivo:", overall_rmse)
for i, rmse in enumerate(per_frame_rmse):
    print(f"Frame {i+1}: {rmse:.3f}")

# %%
for scene_id in range(10):
    overall_rmse, per_frame_rmse = evaluate_tracking_rmse(scene_trackers_results, all_scenes_associations, scene_name=scene_id,nusc=nusc)
    if overall_rmse is not None:
        print("RMSE complessivo scena ",scene_id," :" ,overall_rmse)
        print("RMSE per ciascun frame:")
        for i, rmse in enumerate(per_frame_rmse):
            print(f"Frame {i+1:03d}: {rmse:.3f}")
    else:
        print("  Nessun dato disponibile per la valutazione.")
    print("---------------------------------------\n")

# %%


for scene_id in all_scenes_associations.keys():
    overall_rmse, per_frame_rmse = evaluate_tracking_rmse(
        scene_trackers_results,
        all_scenes_associations,
        scene_name=scene_id,
        nusc=nusc
    )
    if not per_frame_rmse:
        continue

    frames = list(range(1, len(per_frame_rmse) + 1))
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(frames, per_frame_rmse, marker='o')
    ax.set_title(f"Scena {scene_id} — RMSE per frame")
    ax.set_xlabel("Frame")
    ax.set_ylabel("RMSE (px)")
    ax.grid(True)
    fig.tight_layout()

    display(fig)
    plt.close(fig)




