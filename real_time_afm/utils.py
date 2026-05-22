import numpy as np
## change:AE
#import stmpy 
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from skimage.metrics import structural_similarity as ssim
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib.pyplot as plt
import seaborn as sns            
import scipy.ndimage
import torch.nn.functional as F
from models import *
from atomai.utils import get_coord_grid, extract_subimages

def norm_0to1(arr, arr_range = [0, 1]):
    arr = np.asarray(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min()) 
    ## minmax normalization
    arr = arr*(arr_range[1] -arr_range[0])+arr_range[0]
    ## normalization to any defined range
    return arr

def norm_0to1_axis0(arr, arr_range=[0, 1]):
    arr = np.asarray(arr)
    arr_min = arr.min(axis=0, keepdims=True)  
    arr_max = arr.max(axis=0, keepdims=True)  
    arr = (arr - arr_min) / (arr_max - arr_min) 
    arr = arr * (arr_range[1] - arr_range[0]) + arr_range[0]  
    return arr    

def norm_0to1_axis1(arr, arr_range=[0, 1]):
    arr = np.asarray(arr)
    arr_min = arr.min(axis=1, keepdims=True)   # per sample
    arr_max = arr.max(axis=1, keepdims=True)
    arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-8)
    arr_norm = arr_norm * (arr_range[1] - arr_range[0]) + arr_range[0]
    return arr_norm    
    
def bkd_scan(channel_name):
    ch_arr = channel_name.split("_")
    bkd = False
    if ch_arr[-1] == "Bkd":
        bkd = True
    return bkd

def reverse_2D_y(img):
    # Reverses the y_axis of an image. useful to correlate labview coords with real coords
    img_yr = np.zeros(np.shape(img))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_yr[i, j] = img[(img.shape[0]-1)-i, j]            
    return img_yr

def linear_corrected(y):
    y = np.asarray(y)
    x = np.linspace(1,len(y), len(y))
    X = np.asarray(x).reshape((-1, 1))
    reg = LinearRegression(fit_intercept = True).fit(X, y)
    y_corr = y - reg.predict(X)    
    return y_corr

def image_linear_correction(img):
    img = np.asarray(img)
    im1 = []
    im2 = []
    # Linear correction in the horizontal axis
    for line_ind in range(img.shape[0]):
        line_corr =  linear_corrected(img[line_ind])
        im1.append(line_corr)
    im1 = np.asarray(im1)
    # Linear correction in the vertical axis using transpose
    for line_ind in range(im1.T.shape[0]):
        line_corr =  linear_corrected(im1.T[line_ind])
        im2.append(line_corr)
    im2 = np.asarray(im2).T
    return im2
    
def nearest_sample(value, array):
    d = 10000000 + np.max(array) + abs(value)
    ind = 0
    for i in range(len(array)):
        diff = abs(value - array[i])
        if diff <= d:
            ind = i
            d = diff
    return array[ind], ind

def paired_images_spectra_1(image, cits_obj, hyperspectra, window_size = 30, coordinate_step = 10, norm = True):
    """
    Extracts patches from the image and the corresponding spectra from the hyperspectra at the center of each patch.
    Inputs:
        image: 2D numpy array is the morphology
        hyperspectra: 3D numpy array is the hyperspectra
        window_size: int, the width (in pixels) of the patch considered as a feature-example for training
        coordinate_step: int, the distance between the center of two adjacent image patches
    Outputs:
        patches: 3D numpy array. the extracted image patches at index i
        training_spectra: 2D numpy array, the spectrum at the center of each patch at index 1
        coordinates: 2D numpy array, the coordinates of the center of each patch at index i        
    """
    coords = get_coord_grid(image, step = coordinate_step, return_dict= False) 
    # (2704, 2) because at 10 step gap, there will 52x52 points for 512x512 images, and 52x52=2704
    # and 2 is for x,y coordinate
    # print('initial coordinates = ',coords[:, 0].shape)
    extracted_features = extract_subimages(image, coordinates = coords, window_size = window_size)
    # Extract patches (or features) and the center coordinates of each patch.    
    # extracted features has 3 items
    # 0th is the patch images (2304, 30, 30, 1):= 30x30 window size, boundary points are not eligible to have 30x30
    #         so out of 52 points there are 48x48 points eligible = 2304, 30, 30, 1 and 1 for channel grayscale
    # 1th is (2304, 2):= 48x48 or 2304 coordinate of (x, y)
    # 2nd is for donot know, maybe meta data
    patches, coordinates, _ = extracted_features
    patches = patches.squeeze()
    # if norm:
    #     for i in range(len(patches)):
    #         patches[i] = norm_0to1(patches[i])
    n, _, _ = patches.shape
    # total number of pathces that are extracted    
    scan_frame = cits_obj.get_frame_size()
    # scan_frame is the size of the total 512x512 image
    image_pixels = image.shape[0]
    training_spectra = []
    for i in range(len(coordinates)):
        coordinate_point = coordinates[i]*scan_frame/image_pixels
        coord_val, cits_coord = cits_obj.nearest_point(coordinate_point)
        # coordinate point is the point location in the image
        # coord_val is the nearest point location we have spectra for
        # cits_coord is the coordinate of that nearest point in the hyperspectra
        spectra = hyperspectra[cits_coord[0], cits_coord[1], :]
        if norm:
            spectra = norm_0to1(spectra)
            # normalizing
        training_spectra.append(spectra) 
    training_spectra = np.asarray(training_spectra)
    # Reshape the training spectra so that each row is a spectra
    training_spectra = training_spectra.reshape(n, -1)
    # 2304, 121:= 48x48 points and 121 vector for each
    if norm:
        for i in range(len(training_spectra)):
            training_spectra[i] = norm_0to1(training_spectra[i], arr_range = [0, 1])
            # seems redundant as we already used norm_0to1 earlier for these spectra      
    return patches, training_spectra, coordinates
    # (2304, 30, 30), (2304, 121), (2304, 2)


def compute_mse(model, X, y, device, multitask=False):
    """
    Computes Mean Squared Error for each sample in X.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device) 
        if multitask:
            y_pred, _ = model(X_tensor)
        else:
            y_pred = model(X_tensor)
        y_pred = y_pred.cpu().numpy()  
    errors = np.mean((y_pred - y) ** 2, axis=1)  
    return errors

def compute_mse_s2i(model, X, y, device, multitask=False):
    """
    Computes Mean Squared Error for each sample in X.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device) 
        if multitask:
            y_pred, _ = model(X_tensor)
        else:
            y_pred = model(X_tensor)
        y_pred = y_pred.cpu().numpy()  
    errors = np.mean((y_pred - y) ** 2, axis=(1, 2))  
    return errors

def compute_se(model, X, y, device, multitask=False):
    """
    Computes Squared Error for each sample and each feature in y.
    If multitask=True, uses only the spectral prediction from a multitask model.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        if multitask:
            y_pred, _ = model(X_tensor)  # Only use the spectral output
        else:
            y_pred = model(X_tensor)
        y_pred = y_pred.cpu().numpy()
    errors = (y_pred - y) ** 2
    return errors

def compute_se_s2i(model, X, y, device, multitask=False):
    """
    Computes Squared Error for each sample and each feature in y.
    If multitask=True, uses only the spectral prediction from a multitask model.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        if multitask:
            y_pred, _ = model(X_tensor)  # Only use the spectral output
        else:
            y_pred = model(X_tensor)
        y_pred = y_pred.cpu().numpy()
    errors = ((y_pred - y) ** 2).reshape(y.shape[0], -1)
    return errors

def predict_error_model(error_model, X_latent, device, err_multidim=False):
    """
    Predicts error using the trained error model.
    Supports both single-dimensional and multi-dimensional error outputs.
    """
    error_model.eval()
    with torch.no_grad():
        X_latent_tensor = torch.tensor(X_latent, dtype=torch.float32).to(device)
        predicted_errors = error_model(X_latent_tensor).cpu().numpy()
        
        if not err_multidim:  
            predicted_errors = predicted_errors.flatten()  # Convert (N,1) → (N,)
    
    return predicted_errors  # Returns (N, 121) if err_multidim=True, else (N,)

def compute_distances(X_test_latent, X_train_latent):
    """
    Computes the sum of Euclidean distances of each X_test sample 
    from all X_train samples efficiently.
    """
    return np.sum(cdist(X_test_latent, X_train_latent, metric='euclidean'), axis=1)

def compute_sims(X_test_latent):
    """
    Computes the sum of cosine similarities of each X_test sample 
    with all other X_test samples efficiently.
    """
    similarities = cosine_similarity(X_test_latent)  # Uses optimized computation
    return np.sum(similarities, axis=1)

# def acquisition_function(predicted_errors, distances, sims, alpha=1, beta=1, gamma=1,
#                          gp=None, indices=None, coordinates=None, ssim_threshold=0.3):
#     """
#     Computes acquisition function for each test sample.
#     If GP and coordinates are provided, penalize low-SSIM regions based on GP prediction.
#     """
#     base_acq = (alpha * predicted_errors) + (beta * distances) + (gamma * sims)

#     # If using GP-based SSIM filtering
#     if gp is not None and indices is not None and coordinates is not None:
#         # Prepare test coordinates for GP prediction
#         test_coords = np.array([coordinates[i] for i in indices])
#         test_coords_xy = np.stack([test_coords[:, 1], test_coords[:, 0]], axis=1)  # (x, y)

#         # Predict SSIM using GP
#         ssim_pred, _ = gp.predict(test_coords_xy, return_std=True)

#         # Penalize low-SSIM regions
#         adjusted_acq = base_acq.copy()
#         low_ssim_mask = ssim_pred < ssim_threshold
#         adjusted_acq[low_ssim_mask] = base_acq.min()  # or 0

#         return adjusted_acq

#     return base_acq
def acquisition_function(predicted_errors, distances, sims, alpha=1, beta=1, gamma=1,
                         gp=None, indices=None, coordinates=None, r2_threshold=0.8):
    """
    Computes acquisition function for each test sample.
    If GP and coordinates are provided, penalize low-R2 regions based on GP prediction.
    """
    base_acq = (alpha * predicted_errors) + (beta * distances) + (gamma * sims)

    # If using GP-based R2 filtering
    if gp is not None and indices is not None and coordinates is not None:
        # Prepare test coordinates for GP prediction
        test_coords = np.array([coordinates[i] for i in indices])
        test_coords_xy = np.stack([test_coords[:, 1], test_coords[:, 0]], axis=1)  # (x, y)

        # Predict R2 using GP
        r2_pred, _ = gp.predict(test_coords_xy, return_std=True)

        # Penalize low-R2 regions
        adjusted_acq = base_acq.copy()
        low_r2_mask = r2_pred < r2_threshold
        adjusted_acq[low_r2_mask] = base_acq.min()  # or 0

        return adjusted_acq

    return base_acq
    

def set_seed(seed=42):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    ## change:AE    
    #torch.cuda.manual_seed(seed)  # PyTorch GPU
    #torch.cuda.manual_seed_all(seed)  # PyTorch multi-GPU
    #torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    #torch.backends.cudnn.benchmark = False  # May slow down training slightly but ensures reproducibility

def setup_logger(log_filename='outputs/logs/logger.log'):
    logging.basicConfig(
        filename=log_filename,  # Set log file dynamically
        filemode='a',  # Append mode
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO  # Log only INFO and above messages
    )

def log_message(message):
    logging.info(message)    

def plot_spectra_distribution(y_train, y_val, y_test, exp_no='', latent_dim=0, alpha=0, beta=0, gamma=0, r_main=0, step=0):
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(y_train.flatten(), label="Train", fill=True, alpha=0.5)
    sns.kdeplot(y_val.flatten(), label="Validation", fill=True, alpha=0.5)
    sns.kdeplot(y_test.flatten(), label="Test", fill=True, alpha=0.5)
    plt.xlabel("Spectral Intensity")
    plt.ylabel("Density")
    plt.title('Initial Distribution')
    plt.legend()        
    plt.savefig(f'outputs/plots/{exp_no}_spectra_distribution_{latent_dim}_{alpha}_{beta}_{gamma}_{r_main}_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close() 

def plot_image_distribution(y_train, y_val, y_test, exp_no='', latent_dim=0, alpha=0, beta=0, gamma=0, r_main=0, step=0):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(y_train.flatten(), label="Train", fill=True, alpha=0.5)
    sns.kdeplot(y_val.flatten(), label="Validation", fill=True, alpha=0.5)
    sns.kdeplot(y_test.flatten(), label="Test", fill=True, alpha=0.5)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Density")
    plt.title('Initial Pixel Intensity Distribution')
    plt.legend()
    plt.savefig(f'outputs/plots/{exp_no}_image_distribution_{latent_dim}_{alpha}_{beta}_{gamma}_{r_main}_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()    

def plot_spectra_evolution(y_train_active, y_train_random, y_train_active2, y_train_active3, y_train_active4,
                           exp_no='', latent_dim=0, alpha=0, beta=0, gamma=0, r_main=0, step=0):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(y_train_active.flatten(), label=f"Active (im2spec + 1D)", fill=True, alpha=0.5)
    sns.kdeplot(y_train_random.flatten(), label=f"Random", fill=True, alpha=0.5)
    sns.kdeplot(y_train_active2.flatten(), label=f"Active2 (im2spec + 121D)", fill=True, alpha=0.5)
    sns.kdeplot(y_train_active3.flatten(), label=f"Active3 (im2multi + 1D)", fill=True, alpha=0.5)
    sns.kdeplot(y_train_active4.flatten(), label=f"Active4 (im2multi + 121D)", fill=True, alpha=0.5)
    plt.xlabel("Spectral Intensity")
    plt.ylabel("Density")
    plt.title(f"Spectra Distribution at Step {step}")
    plt.legend()
    plt.savefig(f'outputs/plots/{exp_no}_spectra_distribution_{latent_dim}_{alpha}_{beta}_{gamma}_{r_main}_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_model_performance(log_error_active, log_error_active2, log_error_active3, log_error_active4, log_error_random,
                           i, exp_no, latent_dim, alpha, beta, gamma, r_main):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(i+1), log_error_active, label="Active (im2spec + 1D)", marker="s")
    plt.plot(range(i+1), log_error_active2, label="Active2 (im2spec + 121D)", marker="D")
    plt.plot(range(i+1), log_error_active3, label="Active3 (im2multi + 1D)", marker="o")
    plt.plot(range(i+1), log_error_active4, label="Active4 (im2multi + 121D)", marker="P")
    plt.plot(range(i+1), log_error_random, label="Random", marker="x")    
    plt.xlabel("Exploration Step")
    plt.ylabel("Mean Spectral Error")
    plt.legend()
    plt.title("Model Performance Comparison Across Learning Strategies")
    plt.savefig(f'outputs/plots/{exp_no}_error_comparison_{latent_dim}_{alpha}_{beta}_{gamma}_{r_main}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_model_performance_train(
    errors_train_active, errors_train_active2, errors_train_active3, errors_train_active4,
    error_model_active, X_train_active_latent,
    error_model_active2, X_train_active2_latent,
    error_model_active3, X_train_active3_latent,
    error_model_active4, X_train_active4_latent,
    i, exp_no, latent_dim, alpha, beta, gamma, r_main, device
):
    predicted_errors_active = predict_error_model(error_model_active, X_train_active_latent, device)
    predicted_errors_active2 = predict_error_model(error_model_active2, X_train_active2_latent, device, err_multidim=True).mean(axis=1)
    predicted_errors_active3 = predict_error_model(error_model_active3, X_train_active3_latent, device)
    predicted_errors_active4 = predict_error_model(error_model_active4, X_train_active4_latent, device, err_multidim=True).mean(axis=1)

    errors_train_active2 = errors_train_active2.mean(axis=1)
    errors_train_active4 = errors_train_active4.mean(axis=1)

    plt.figure(figsize=(16, 10), dpi=300)

    titles = ["Active (im2spec + 1D)", "Active2 (im2spec + 121D)", "Active3 (im2multi + 1D)", "Active4 (im2multi + 121D)"]
    errors_true = [errors_train_active, errors_train_active2, errors_train_active3, errors_train_active4]
    errors_pred = [predicted_errors_active, predicted_errors_active2, predicted_errors_active3, predicted_errors_active4]
    colors = ['blue', 'green', 'orange', 'purple']

    for idx in range(4):
        plt.subplot(2, 2, idx + 1)
        plt.scatter(errors_true[idx], errors_pred[idx], alpha=0.5, color=colors[idx])
        plt.plot([min(errors_true[idx]), max(errors_true[idx])],
                 [min(errors_true[idx]), max(errors_true[idx])], 'r--')
        plt.xlabel("Ground Truth Error (Per Sample)")
        plt.ylabel("Predicted Error (Per Sample)")
        plt.title(f"{titles[idx]} - Step {i}")

    plt.tight_layout()
    plt.savefig(f'outputs/plots/{exp_no}_error_model_performance_train_{latent_dim}_{alpha}_{beta}_{gamma}_{r_main}_step_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_model_performance(
    errors_test_active, errors_test_active2, errors_test_active3, errors_test_active4,
    error_model_active, X_test_active_latent, error_model_active2, X_test_active2_latent,
    error_model_active3, X_test_active3_latent, error_model_active4, X_test_active4_latent,
    i, exp_no, latent_dim, alpha, beta, gamma, r_main, device
):

    predicted_errors_active = predict_error_model(error_model_active, X_test_active_latent, device)
    predicted_errors_active2 = predict_error_model(error_model_active2, X_test_active2_latent, device, err_multidim=True).mean(axis=1)
    predicted_errors_active3 = predict_error_model(error_model_active3, X_test_active3_latent, device)
    predicted_errors_active4 = predict_error_model(error_model_active4, X_test_active4_latent, device, err_multidim=True).mean(axis=1)

    errors_test_active2 = errors_test_active2.mean(axis=1)
    errors_test_active4 = errors_test_active4.mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    axes[0, 0].scatter(errors_test_active, predicted_errors_active, alpha=0.5, color='blue')
    axes[0, 0].plot([errors_test_active.min(), errors_test_active.max()],
                   [errors_test_active.min(), errors_test_active.max()], "r--")
    axes[0, 0].set_title(f"Active (im2spec + 1D) - Step {i}")
    axes[0, 0].set_xlabel("Ground Truth Error")
    axes[0, 0].set_ylabel("Predicted Error")

    axes[0, 1].scatter(errors_test_active2, predicted_errors_active2, alpha=0.5, color='green')
    axes[0, 1].plot([errors_test_active2.min(), errors_test_active2.max()],
                   [errors_test_active2.min(), errors_test_active2.max()], "r--")
    axes[0, 1].set_title(f"Active2 (im2spec + 121D) - Step {i}")
    axes[0, 1].set_xlabel("Ground Truth Error")
    axes[0, 1].set_ylabel("Predicted Error")

    axes[1, 0].scatter(errors_test_active3, predicted_errors_active3, alpha=0.5, color='orange')
    axes[1, 0].plot([errors_test_active3.min(), errors_test_active3.max()],
                   [errors_test_active3.min(), errors_test_active3.max()], "r--")
    axes[1, 0].set_title(f"Active3 (im2multi + 1D) - Step {i}")
    axes[1, 0].set_xlabel("Ground Truth Error")
    axes[1, 0].set_ylabel("Predicted Error")

    axes[1, 1].scatter(errors_test_active4, predicted_errors_active4, alpha=0.5, color='purple')
    axes[1, 1].plot([errors_test_active4.min(), errors_test_active4.max()],
                   [errors_test_active4.min(), errors_test_active4.max()], "r--")
    axes[1, 1].set_title(f"Active4 (im2multi + 121D) - Step {i}")
    axes[1, 1].set_xlabel("Ground Truth Error")
    axes[1, 1].set_ylabel("Predicted Error")

    plt.tight_layout()
    plt.savefig(f'outputs/plots/{exp_no}_error_model_performance_{latent_dim}_{alpha}_{beta}_{gamma}_{r_main}_step_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_selected_samples(
    full_image, coordinates, 
    coordinates_active_selected, coordinates_active2_selected, coordinates_active3_selected, coordinates_active4_selected, coordinates_random_selected,
    i, exp_no, latent_dim, alpha, beta, gamma, r_main
):

    plt.figure(figsize=(8, 8), dpi=300)

    # Show the morphology image
    plt.imshow(full_image, cmap='gray')

    # Overlay cumulative selected points for each strategy
    plt.scatter(coordinates_active_selected[:, 0], coordinates_active_selected[:, 1],
                c='blue', label="Active (im2spec + 1D)", marker="s", edgecolors="white", s=80, alpha=0.7)

    plt.scatter(coordinates_active2_selected[:, 0], coordinates_active2_selected[:, 1],
                c='green', label="Active2 (im2spec + 121D)", marker="D", edgecolors="white", s=80, alpha=0.7)

    plt.scatter(coordinates_active3_selected[:, 0], coordinates_active3_selected[:, 1],
                c='orange', label="Active3 (im2multi + 1D)", marker="o", edgecolors="white", s=80, alpha=0.7)

    plt.scatter(coordinates_active4_selected[:, 0], coordinates_active4_selected[:, 1],
                c='purple', label="Active4 (im2multi + 121D)", marker="P", edgecolors="white", s=80, alpha=0.7)

    plt.scatter(coordinates_random_selected[:, 0], coordinates_random_selected[:, 1],
                c='red', label="Random", marker="x", s=80, alpha=0.7)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(f"Selected Samples at Step {i}")
    plt.legend()
    # Save plot
    plt.savefig(f'outputs/plots/{exp_no}_selected_samples_{latent_dim}_{alpha}_{beta}_{gamma}_{r_main}_step_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Free memory

    
def plot_acquisition_heatmap(full_image, 
                             coordinates_active, acquisition_values_active, 
                             coordinates_active2, acquisition_values_active2,
                             coordinates_active3, acquisition_values_active3,
                             coordinates_active4, acquisition_values_active4,
                             i, exp_no, latent_dim, alpha, beta, gamma, r_main):
    
    plt.figure(figsize=(16, 10), dpi=300)

    # Create empty heatmaps with the same size as full_image
    heatmap_active = np.zeros_like(full_image, dtype=np.float32)
    heatmap_active2 = np.zeros_like(full_image, dtype=np.float32)
    heatmap_active3 = np.zeros_like(full_image, dtype=np.float32)
    heatmap_active4 = np.zeros_like(full_image, dtype=np.float32)

    # Assign acquisition scores to their respective coordinates
    for j, coord in enumerate(coordinates_active):
        x, y = int(coord[0]), int(coord[1])
        heatmap_active[y, x] = acquisition_values_active[j]

    for j, coord in enumerate(coordinates_active2):
        x, y = int(coord[0]), int(coord[1])
        heatmap_active2[y, x] = acquisition_values_active2[j]

    for j, coord in enumerate(coordinates_active3):
        x, y = int(coord[0]), int(coord[1])
        heatmap_active3[y, x] = acquisition_values_active3[j]

    for j, coord in enumerate(coordinates_active4):
        x, y = int(coord[0]), int(coord[1])
        heatmap_active4[y, x] = acquisition_values_active4[j]

    # Apply Gaussian smoothing for better visualization
    heatmap_active = scipy.ndimage.gaussian_filter(heatmap_active, sigma=2)
    heatmap_active2 = scipy.ndimage.gaussian_filter(heatmap_active2, sigma=2)
    heatmap_active3 = scipy.ndimage.gaussian_filter(heatmap_active3, sigma=2)
    heatmap_active4 = scipy.ndimage.gaussian_filter(heatmap_active4, sigma=2)

    # Plot in a 2x2 grid
    titles = ["Active (im2spec + 1D)", "Active2 (im2spec + 121D)",
              "Active3 (im2multi + 1D)", "Active4 (im2multi + 121D)"]
    heatmaps = [heatmap_active, heatmap_active2, heatmap_active3, heatmap_active4]

    for idx, heatmap in enumerate(heatmaps):
        plt.subplot(2, 2, idx + 1)
        plt.imshow(full_image, cmap="gray", alpha=0.5)
        plt.imshow(heatmap, cmap="jet", alpha=0.6)
        plt.colorbar(label="Acquisition Score")
        plt.title(f"{titles[idx]} (Step {i})")

    plt.tight_layout()
    plt.savefig(f'outputs/plots/{exp_no}_acquisition_heatmap_{latent_dim}_{alpha}_{beta}_{gamma}_{r_main}_step_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_acquisition_histogram(acquisition_values_active,
                               acquisition_values_active2,
                               acquisition_values_active3,
                               acquisition_values_active4,
                               i, exp_no, latent_dim, alpha, beta, gamma, r_main):
    
    plt.figure(figsize=(16, 10), dpi=300)

    bins = 30
    alpha_val = 0.7

    # Plot histogram for Active
    plt.subplot(2, 2, 1)
    plt.hist(acquisition_values_active, bins=bins, color='blue', alpha=alpha_val)
    plt.xlabel("Acquisition Score")
    plt.ylabel("Frequency")
    plt.title(f"Active (im2spec + 1D) - Step {i}")

    # Plot histogram for Active2
    plt.subplot(2, 2, 2)
    plt.hist(acquisition_values_active2, bins=bins, color='green', alpha=alpha_val)
    plt.xlabel("Acquisition Score")
    plt.ylabel("Frequency")
    plt.title(f"Active2 (im2spec + 121D) - Step {i}")

    # Plot histogram for Active3
    plt.subplot(2, 2, 3)
    plt.hist(acquisition_values_active3, bins=bins, color='purple', alpha=alpha_val)
    plt.xlabel("Acquisition Score")
    plt.ylabel("Frequency")
    plt.title(f"Active3 (im2multi + 1D) - Step {i}")

    # Plot histogram for Active4
    plt.subplot(2, 2, 4)
    plt.hist(acquisition_values_active4, bins=bins, color='orange', alpha=alpha_val)
    plt.xlabel("Acquisition Score")
    plt.ylabel("Frequency")
    plt.title(f"Active4 (im2multi + 121D) - Step {i}")

    plt.tight_layout()
    plt.savefig(f'outputs/plots/{exp_no}_acquisition_histogram_{latent_dim}_{alpha}_{beta}_{gamma}_{r_main}_step_{i}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_true_vs_pred(model, X, y, device, title, num_samples=10):
    """
    Plots the true vs. predicted spectra for the first 'num_samples' samples.
    Each target spectrum has 121 values.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X[:num_samples], dtype=torch.float32).to(device)
        y_pred = model(X_tensor).cpu().numpy()  # Get predictions

    fig, axes = plt.subplots(num_samples, 1, figsize=(8, 2 * num_samples), sharex=True)

    if num_samples == 1:
        axes = [axes]  # Ensure axes is iterable when only one sample

    for i in range(num_samples):
        ax = axes[i]
        ax.plot(y[i], label="True Spectrum", linestyle="dashed", color="blue")
        ax.plot(y_pred[i], label="Predicted Spectrum", alpha=0.7, color="red")
        ax.set_title(f"Sample {i+1}")
        ax.set_ylabel("Intensity")
        ax.legend()

    plt.suptitle(title)
    plt.xlabel("Spectral Index (0-120)")
    plt.tight_layout()
    plt.show()

def plot_true_vs_pred_errors(error_model, X_latent, true_errors, device, title, multidim=False):
    """
    Plots true vs. predicted errors for the given dataset.
    """
    error_model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_latent, dtype=torch.float32).to(device)
        pred_errors = error_model(X_tensor).cpu().numpy()  # Get predictions

    if multidim:
        true_errors = true_errors.mean(axis=1)
        pred_errors = pred_errors.mean(axis=1)  # Ensure correct shape
    else:
        true_errors = true_errors.flatten()
        pred_errors = pred_errors.flatten()
    
    # Scatter plot: True vs Predicted Errors
    plt.figure(figsize=(4, 4))
    plt.scatter(true_errors, pred_errors, alpha=0.6, label="Samples")
    plt.plot([min(true_errors), max(true_errors)], [min(true_errors), max(true_errors)], 'r--', label="Ideal Fit")  
    plt.xlabel("True Error")
    plt.ylabel("Predicted Error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def BEPS_image_spectral_pairs(beps_file_path, window_size = 16, step = 1):
    input_file = np.load(beps_file_path)
    image = input_file['image']
    #print(image.shape)
    spectra = input_file['spectra']
    #print(spectra.shape)
    v_step = input_file['spec_step_vol']
    # Extract patches
    coordinates = get_coord_grid(image, step = step, return_dict=False) ## returns an array of [y, x] locations where patches will be extracted centering [y, x]
    # extract image patch for each point on a grid
    window_size = window_size
    features_all, coords, _ = extract_subimages(image, coordinates, window_size) ## extract image patches centered at coordinates
    patches = features_all[:,:,:,0]
    indices_all = np.array(coords, dtype = int)
    # extract spectra 
    n = patches.shape[0]
    all_spectra = []
    for ind in range(n):
        spectrum =  spectra[indices_all[ind,0], indices_all[ind,1]]  ## spectra uses [y, x] indexing -> indices_all also comes from coords that uses [y, x] -> so no need to swap index
        all_spectra.append(spectrum)
    all_spectra = np.array(all_spectra)
    ## patches are window size x window size
    ## spectra are spectral responses taken at the center point of each patch
    ## indices are [y, x] == [row, col] coordinate of the center of each patch in original full image 
    return patches, all_spectra, indices_all, v_step

def extract_beps_data(beps_file_path):
    input_file = np.load(beps_file_path)    
    image = input_file['image']
    #print(image.shape)
    spectra = input_file['spectra']
    #print(spectra.shape)
    v_step = input_file['spec_step_vol']
    return image, spectra, v_step

def train_im2spec(model, train_data, val_data, device, num_epochs=1000, lr=1e-3, batch_size=32, wdecay=1e-4, wloss=2, beta=0.05):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if val_data:
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    

    def weighted_mse_loss(pred, target, x=2):  # x is the exponent parameter
        weights = target ** x  # Higher target → More weight
        return torch.mean(weights * (pred - target) ** 2)
        
    def weighted_smooth_l1_loss(pred, target, beta=0.1, x=2):
        """
        Weighted Smooth L1 Loss (Huber Loss) that gives higher weight to larger targets.
        """
        weights = target ** x  # Higher target → More weight
        huber_loss = torch.where(
            torch.abs(pred - target) < beta,
            0.5 * (pred - target) ** 2 / beta,  # Quadratic region (MSE-like)
            torch.abs(pred - target) - 0.5 * beta  # L1 region
        )
        # print(target.shape, pred.shape) ## [16, 121] ==> the weight has also a dimension of [16, 121]     
        return torch.mean(weights * huber_loss)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
    # criterion = lambda pred, target: weighted_mse_loss(pred, target, x=wloss) ## nn.MSELoss() ## nn.L1Loss()
    criterion = lambda pred, target: weighted_smooth_l1_loss(pred, target, beta=beta, x=wloss)
    

    ## linear decay Learning Rate Scheduler (starts at lr, reduces to 0.1 * lr at last epoch)
    lr_lambda = lambda epoch: 1 - (epoch / num_epochs) * 0.9  # Reduces from lr → 0.1 * lr
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_loss = float("inf")
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            if batch_x.size(0) == 1:
                continue                        
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation Step
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    preds = model(batch_x)
                    loss = criterion(preds, batch_y)
                    val_loss += loss.item()
    
            val_loss /= len(val_loader)
    
            # Adjust Learning Rate
            scheduler.step()
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
    
            if epoch%(num_epochs//4)==0:
                log_message(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    # Load the best model state before returning
    if val_loader:
        model.load_state_dict(best_model_state)
    return model

    
def train_spec2im(model, train_data, val_data, device, num_epochs=1000, lr=1e-3, batch_size=32, wdecay=1e-4, wloss=1, beta=0.05):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    def weighted_smooth_l1_loss(pred, target, beta=0.1, x=1):
        weights = target ** x
        huber_loss = torch.where(
            torch.abs(pred - target) < beta,
            0.5 * (pred - target) ** 2 / beta,
            torch.abs(pred - target) - 0.5 * beta
        )
        return torch.mean(weights * huber_loss)

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
    criterion = lambda pred, target: weighted_smooth_l1_loss(pred, target, beta=beta, x=wloss)

    # Linear LR scheduler
    lr_lambda = lambda epoch: 1 - (epoch / num_epochs) * 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            if batch_x.size(0) == 1:
                continue                                    
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if epoch % (num_epochs // 4) == 0:
            log_message(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    model.load_state_dict(best_model_state)
    return model

def train_im2multi(model, train_data, val_data, device,
                   num_epochs=1000, lr=1e-3, batch_size=32,
                   wdecay=1e-4, wloss=2, beta=0.05, r_main=0.5, validation=1):
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    def weighted_mse_loss(pred, target, x=2):  # x is the exponent parameter
        weights = target ** x  # Higher target → More weight
        return torch.mean(weights * (pred - target) ** 2)
        
    def weighted_smooth_l1_loss(pred, target, beta=0.1, x=2):
        """
        Weighted Smooth L1 Loss (Huber Loss) that gives higher weight to larger targets.
        """
        weights = target ** x  # Higher target → More weight
        huber_loss = torch.where(
            torch.abs(pred - target) < beta,
            0.5 * (pred - target) ** 2 / beta,  # Quadratic region (MSE-like)
            torch.abs(pred - target) - 0.5 * beta  # L1 region
        )
        return torch.mean(weights * huber_loss)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
    # criterion = lambda pred, target: weighted_mse_loss(pred, target, x=wloss) ## nn.MSELoss() ## nn.L1Loss()
    criterion = lambda pred, target: weighted_smooth_l1_loss(pred, target, beta=beta, x=wloss)

    ## linear decay Learning Rate Scheduler (starts at lr, reduces to 0.1 * lr at last epoch)    
    lr_lambda = lambda epoch: 1 - (epoch / num_epochs) * 0.9  # Reduces from lr → 0.1 * lr
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            if batch_x.size(0) == 1:
                continue                                    
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred_spectra, recon_img = model(batch_x)
            # log_message(f': {batch_x.shape}, {batch_y.shape}, {pred_spectra.shape}, {recon_img.shape}')

            loss_spectra = criterion(pred_spectra, batch_y)
            loss_image = F.l1_loss(recon_img, batch_x)  # Unweighted image reconstruction loss
            loss = (r_main * loss_spectra) + ((1-r_main) * loss_image)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_spectra, recon_img = model(batch_x)
                loss_spectra = criterion(pred_spectra, batch_y)
                loss_image = F.l1_loss(recon_img, batch_x)
                if validation == 1:
                    loss = loss_spectra ## (r_main * loss_spectra) + ((1-r_main) * loss_image)
                elif validation == 2:
                    loss = (r_main * loss_spectra) + ((1-r_main) * loss_image)
                else:
                    loss = loss_spectra                     
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if epoch % (num_epochs // 4) == 0:
            log_message(f'[Multi] Epoch [{epoch+1}/{num_epochs}], '
                        f'Train Loss: {train_loss:.4f}, '
                        f'Val Loss: {val_loss:.4f}, '
                        f'LR: {scheduler.get_last_lr()[0]:.6f}')

    model.load_state_dict(best_model_state)
    return model

def train_spec2multi(model, train_data, val_data, device,
                     num_epochs=1000, lr=1e-3, batch_size=32,
                     wdecay=1e-4, wloss=1, beta=0.05, r_main=0.5, validation=1):
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    def weighted_smooth_l1_loss(pred, target, beta=0.1, x=1):
        weights = target ** x
        huber_loss = torch.where(
            torch.abs(pred - target) < beta,
            0.5 * (pred - target) ** 2 / beta,
            torch.abs(pred - target) - 0.5 * beta
        )
        return torch.mean(weights * huber_loss)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
    criterion = lambda pred, target: weighted_smooth_l1_loss(pred, target, beta=beta, x=wloss)

    lr_lambda = lambda epoch: 1 - (epoch / num_epochs) * 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            if batch_x.size(0) == 1:
                continue                                    
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred_img, recon_spec = model(batch_x)

            loss_img = criterion(pred_img, batch_y)             # predicted image vs ground truth
            loss_spec = F.l1_loss(recon_spec, batch_x)          # reconstructed input spectra vs original input
            loss = (r_main * loss_img) + ((1 - r_main) * loss_spec)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_img, recon_spec = model(batch_x)

                loss_img = criterion(pred_img, batch_y)
                loss_spec = F.l1_loss(recon_spec, batch_x)

                if validation == 1:
                    loss = loss_img
                elif validation == 2:
                    loss = (r_main * loss_img) + ((1 - r_main) * loss_spec)
                else:
                    loss = loss_img
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if epoch % (num_epochs // 4) == 0:
            log_message(f'[Spec2Multi] Epoch [{epoch+1}/{num_epochs}], '
                        f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                        f'LR: {scheduler.get_last_lr()[0]:.6f}')

    model.load_state_dict(best_model_state)
    return model

def train_error_model(model, train_data, val_data, device, num_epochs=1000, lr=1e-3, batch_size=32, wdecay=1e-4, wloss=2, beta=0.05):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if val_data:
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    def weighted_mse_loss(pred, target, x=2):  # x is the exponent parameter
        weights = target ** x  # Higher target → More weight
        return torch.mean(weights * (pred - target) ** 2)

    def weighted_smooth_l1_loss(pred, target, beta=0.1, x=2):
        
        # #################################################################        
        # """
        # Weighted Smooth L1 Loss (Huber Loss)
        # - If target has shape [B, 121], apply small weights to first 20 dims
        # - If target has shape [B, 1], apply weight = 1
        # """
        # # Multi-dimensional case (e.g., [B, 121])
        # if target.ndim == 2 and target.shape[1] > 1:
        #     dim_weights = torch.cat([
        #         torch.full((20,), 1e-4, device=target.device),
        #         torch.ones((target.shape[1] - 20,), device=target.device)
        #     ])  # shape: [121]
        #     weights = dim_weights.unsqueeze(0).expand_as(target)  # shape: [B, 121]
        # else:
        #     # 1D case (e.g., [B, 1])
        #     weights = torch.ones_like(target)  # or simply: weights = torch.ones_like(target)
        # #################################################################

        abs_diff = torch.abs(pred - target)        
        weights = torch.full_like(target, fill_value=x)
        huber_loss = torch.where(
            abs_diff < beta,
            0.5 * abs_diff ** 2 / beta,
            abs_diff - 0.5 * beta
        )
        return torch.mean(weights * huber_loss)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
    # criterion = lambda pred, target: weighted_mse_loss(pred, target, x=wloss) ## nn.MSELoss() ## nn.L1Loss()
    criterion = lambda pred, target: weighted_smooth_l1_loss(pred, target, beta=beta, x=wloss)
    
    

    ## linear decay Learning Rate Scheduler (starts at lr, reduces to 0.1 * lr at last epoch)
    lr_lambda = lambda epoch: 1 - (epoch / num_epochs) * 0.9  # Reduces from lr → 0.1 * lr
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_loss = float("inf")
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            if batch_x.size(0) == 1:
                continue                                    
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation Step
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    preds = model(batch_x)
                    loss = criterion(preds, batch_y)
                    val_loss += loss.item()
    
            val_loss /= len(val_loader)
    
            # Adjust Learning Rate
            scheduler.step()
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
            if epoch%(num_epochs//4)==0:
                log_message(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    # Load the best model state before returning
    if val_loader:
        model.load_state_dict(best_model_state)
    return model


def objective_im2spec(trial, dim_in, dim_out, latent_dim, device, train_dataset, val_dataset, num_epochs, lr, batch_size):
    # hp
    nb_filters_enc = trial.suggest_categorical("nb_filters_enc", [32, 64, 128])
    nb_filters_dec = trial.suggest_categorical("nb_filters_dec", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    wdecay = trial.suggest_float("wdecay", 1e-6, 1e-3, log=True)
    wloss = trial.suggest_int("wloss", 2, 2)
    beta = trial.suggest_float("beta", 0.01, 0.2)
    # model
    model = im2spec(
        feature_size=dim_in,
        target_size=dim_out,
        latent_dim=latent_dim,
        nb_filters_enc=nb_filters_enc,
        nb_filters_dec=nb_filters_dec,
        dropout=dropout
    ).to(device)
    # train
    model = train_im2spec(
        model,
        train_dataset,
        val_dataset,
        device,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        wdecay=wdecay,
        wloss=wloss,
        beta=beta
    )
    # val
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            loss = F.mse_loss(preds, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    trial.set_user_attr("trained_model", model)    
    return val_loss

    
def objective_spec2im(trial, dim_in, dim_out, latent_dim, device, train_dataset, val_dataset, num_epochs, lr, batch_size):
    # Suggested hyperparameters
    nb_filters_enc = trial.suggest_categorical("nb_filters_enc", [32, 64, 128])
    nb_filters_dec = trial.suggest_categorical("nb_filters_dec", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    wdecay = trial.suggest_float("wdecay", 1e-6, 1e-3, log=True)
    wloss = trial.suggest_int("wloss", 1, 1)
    beta = trial.suggest_float("beta", 0.01, 0.2)

    # Model instantiation
    model = spec2im(
        feature_size=dim_in,
        target_size=dim_out,
        latent_dim=latent_dim,
        nb_filters_enc=nb_filters_enc,
        nb_filters_dec=nb_filters_dec,
        dropout=dropout
    ).to(device)

    # Train model
    model = train_spec2im(  # You’ll need to implement this function similar to `train_im2spec`
        model,
        train_dataset,
        val_dataset,
        device,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        wdecay=wdecay,
        wloss=wloss,
        beta=beta
    )

    # Validation loss
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            loss = F.mse_loss(preds, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    trial.set_user_attr("trained_model", model)
    return val_loss

def objective_error_model(
    trial,
    latent_dim,
    device,
    train_dataset,
    val_dataset,
    num_epochs=1000,
    lr=1e-3,
    batch_size=32,
):
    # hp
    conv1_channels = trial.suggest_categorical("conv1_channels", [16, 32, 64])
    conv2_channels = trial.suggest_categorical("conv2_channels", [32, 64, 128])
    wdecay = trial.suggest_float("wdecay", 1e-6, 1e-3, log=True)
    wloss = trial.suggest_int("wloss", 1, 1)  ## 
    beta = trial.suggest_float("beta", 0.01, 0.2)
    # model
    model = ErrorModel(
        latent_dim=latent_dim,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels
    ).to(device)
    # train
    trained_model = train_error_model(
        model,
        train_dataset,
        val_dataset,
        device,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        wdecay=wdecay,
        wloss=wloss,
        beta=beta
    )
    # val
    trained_model.eval()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = trained_model(batch_x)
            loss = F.mse_loss(preds, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    trial.set_user_attr("trained_model", trained_model)
    return val_loss

class SpectraDataset(Dataset):
    """Custom PyTorch dataset for image-spectra pairs."""
    def __init__(self, images, spectra):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.spectra = torch.tensor(spectra, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.spectra[idx]

class ImageDataset(Dataset):
    """Custom PyTorch dataset for spectra-to-image pairs."""
    def __init__(self, spectra, images):
        self.spectra = torch.tensor(spectra, dtype=torch.float32)  # input
        self.images = torch.tensor(images, dtype=torch.float32)    # target

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx], self.images[idx]

class ErrorDataset(Dataset):
    """
    Dataset for training the error model. Uses latent representations from im2spec.
    """
    def __init__(self, latent_features, errors, err_multidim=False):
        self.latent_features = torch.tensor(latent_features, dtype=torch.float32)
        if not err_multidim:
            self.errors = torch.tensor(errors, dtype=torch.float32).unsqueeze(1)  # Make it shape (N, 1)
        else:
            self.errors = torch.tensor(errors, dtype=torch.float32)  # No unsqueeze needed
            
    def __len__(self):
        return len(self.errors)

    def __getitem__(self, idx):
        return self.latent_features[idx], self.errors[idx]

class Sxm_Image():
    
    def __init__(self, file_name):
        self.file = stmpy.load(str(file_name))
        # Metadata of the sxm file
        self.header = self.file.header
        # Size of the scan frame in (m)
        self.frame = self.file.header.get('scan_range')[0]
        # Pixels in an image
        self.pixels = self.file.header.get('scan_pixels')
        self.scan_offset = self.file.header.get('scan_offset')
        self.scan_angle = self.file.header.get('scan_angle')
        self.scan_dir =  self.file.header.get('scan_dir')

    def image(self, channel = "Z_Fwd", linear_correction = True): # If nothing is provided the default channel is "ZFwd"
        ''' Converts the sxm file to a 2D image array
            In the absence of argument, default channel is "Z_Fwd"
        '''
        image  = self.file.channels[channel]  
        # linear baseline correction of the image
        if linear_correction == True:
            image = image_linear_correction(image)
        # correcting the y-coords that are reversed for scan_dir = 'down'
        if self.scan_dir == 'down':
            image = reverse_2D_y(image)
        if bkd_scan(channel) == True:
            image = reverse_2D_x(image)
        return image

    def get_channels(self):
        ''' Outputs the channel names in the sxm file
        ''' 
        channel_names = []
        for key in self.file.channels:
            channel_names.append(key)
        return channel_names

class CITS_Analysis():
    #import stmpy
    """Functions to extract and analyze CITS data from .3ds file"""
    def __init__(self, filename):
        self.biasOffset = False
        #self.data = str(filename)
        smd = stmpy.load(filename, biasOffset = self.biasOffset)
        self.data = smd
        self.header = smd.header
        self.V_range = np.asarray(smd.en)
        self.data_size = smd.header["Grid dim"]  
                                                      
        def rearrange_for_spectrum(array_3d):
            '''
            This rearranges the hyperspectral data to assert the spectrum as the third index.
            The initial two index are the position index
            
            '''
            a1 = np.zeros((array_3d.shape[1], array_3d.shape[2], array_3d.shape[0]))
            for i in range(array_3d.shape[1]):
                for j in range(array_3d.shape[2]):
                    #print(self.current[i, j])
                    a1[i, j, :] = array_3d[:, i, j] 
            return np.asarray(a1)
        
        self.current = rearrange_for_spectrum(smd.I)     
        self.didv_x = rearrange_for_spectrum(smd.grid['LIX 1 omega (A)'])
        self.didv_y = rearrange_for_spectrum(smd.grid['LIY 1 omega (A)'])
        
    def get_frame_size(self):
        d_line = self.header["Grid settings"]
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        result = [float(x) for x in re.findall(match_number, d_line)]
        return result[2], result[3]
        
    def nearest_V(self, value):
        V_val, V_ind = nearest_sample(value, self.V_range)
        return V_val, V_ind

    def nearest_point(self, coord):
        x_vector = np.linspace(0, self.get_frame_size()[0], self.current.shape[0])
        y_vector = np.linspace(0, self.get_frame_size()[1], self.current.shape[1])
        x_val, x_ind = nearest_sample(coord[0], x_vector)
        y_val, y_ind = nearest_sample(coord[1], y_vector)
        return [x_val, y_val], [x_ind, y_ind]

    def current_map(self, voltage):
        v_actual, v_ind = self.nearest_V(voltage)
        return self.current[:, :, v_ind], v_actual        

    def didv_x_map(self, voltage):
        v_actual, v_ind = self.nearest_V(voltage)
        return self.didv_x[:, :, v_ind], v_actual  

        
