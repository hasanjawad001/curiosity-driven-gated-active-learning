if __name__=='__main__':
    import numpy as np
    import os
    
    exp_no = 'rebuttal_3b'
    
    data = np.load(f"outputs/2c_correct_save_data.npz")
    image = data["img"]
    spec_step_vol = data["vdc_vec2"]
    spectra = data["pola_off_field2"]
    r2_score = data["r2_off_field_mean"]
    
    print(image.shape, spec_step_vol.shape, spectra.shape, r2_score.shape)
    
    data_noise = np.load(f"outputs/1a_noise_info.npz")
    noisy_indices = data_noise['noisy_indices']
    noisy_score = data_noise['noisy_score']
    noisy_map = data_noise['noisy_map']
    
    print(noisy_indices.shape, noisy_score.shape, noisy_map.shape)
    
    # for folder in [
    #     "outputs/logs",
    #     "outputs/models",
    #     "outputs/paper",
    #     "outputs/plots"
    # ]:
    #     os.makedirs(folder, exist_ok=True)
    # np.savez_compressed(
    #     "inputs/data/noisy_data.npz",
    #     image = image,
    #     spec_step_vol = spec_step_vol,
    #     spectra = spectra,
    #     r2_score = r2_score,
    #     noisy_map = noisy_map,    
    #     noisy_indices = noisy_indices,
    #     noisy_score = noisy_score,
    # )
    
    from atomai.utils import get_coord_grid, extract_patches_and_spectra, extract_patches, extract_subimages
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, ListedColormap
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time
    from typing import List, Tuple
    import random
    from torch.utils.data import DataLoader, Dataset, random_split
    import optuna
    from utils import *
    from models import *
    import copy
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    import os
    import pickle
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    # !pip install scipy
    from scipy.stats import qmc
    import matplotlib.patches as patches
    import pickle

    n_trials = 30 ## 30
    exploration_steps = 26 ## 26  ## int((total_samples * r_test)/num_sample_exp) - 1 ## 100 
    terminate_check = False
    d_result = {}
    d_ratios = {}
    for trial_no in range(n_trials):
        seed = int(42 + trial_no)    
        setup_logger(f'outputs/logs/{exp_no}_logger.log') 
        log_message(f'=======================================')
        log_message(f'============== {exp_no} ===============')
        log_message(f'=======================================')
        set_seed(seed) 
        rng = np.random.default_rng(seed=seed)
        list_opt = [ 
            (0.90, 0.05, 0.05, 0.50), ## err, dist, sim, spectralPred          
        ]
        opt = list_opt[0]
        log_message(f'===== {opt} =====')
        dir_prefix = 'inputs/'
        
        ##
        ## 1
        beps_file = r"" + dir_prefix + "data/noisy_data.npz"
        full_image, spectra, v_step = extract_beps_data(beps_file)
        log_message(f'full_image.shape: {full_image.shape}, spectra.shape: {spectra.shape}, v_step.shape: {v_step.shape}')
        print(f'full_image.shape: {full_image.shape}, spectra.shape: {spectra.shape}, v_step.shape: {v_step.shape}')
        print(np.min(full_image), np.max(full_image), np.min(spectra), np.max(spectra))
        
        ##
        ## 2a
        images, spectra, coordinates, vstep = BEPS_image_spectral_pairs(beps_file, window_size = 4)
        y, X = norm_0to1(images), norm_0to1_axis1(spectra)
        log_message(f'X.shape: {X.shape}, y.shape: {y.shape}, coordinates.shape: {coordinates.shape}, vstep.shape: {vstep.shape}')
        print(f'X.shape: {X.shape}, y.shape: {y.shape}, coordinates.shape: {coordinates.shape}, vstep.shape: {vstep.shape}')
        print(np.min(images), np.max(images), np.min(spectra), np.max(spectra))
        print(np.min(X), np.max(X), np.min(y), np.max(y))
        
        ##
        ## 3a
        dim_in = len(X[0]) # Input dimensions (spectra length)
        dim_out = y[0].shape  # Output dimensions (image height and width)
        log_message(f'dim_in: {dim_in}, dim_out: {dim_out}')
        print(f'dim_in: {dim_in}, dim_out: {dim_out}')
        indices = np.arange(len(X))
        log_message(f'indices.shape: {indices.shape}, indices[-1]: {indices[-1]}')
        print(f'indices.shape: {indices.shape}, indices[-1]: {indices[-1]}')
        
        ## 0
        # obj_n_trial = 1 ## >= 10
        r_test = 0.9 ## (train+val) to test
        r_val = 0.9 ## train to val
        ## changeR
        r2_thresholdA = 0.02
        r2_thresholdB = 0.22
        r2_thresholdC = 0.42
        r2_thresholdD = 0.62
        r2_thresholdE = 0.82

        init_lr = 1e-3 ## 1e-3 
        init_num_epochs = 100 ## 100
        total_samples = int(X.shape[0])
        batch_size = int(total_samples * 0.02) ## 16 
        latent_dim = int(batch_size * 0.5) ## 32
        num_sample_exp = int(total_samples * 0.005) ## 10  

        alpha = opt[0] ## exploit error model?
        beta = opt[1] ## explore distant samples?
        gamma = opt[2] ## explore high representative samples?
        r_main = opt[3] ## weigh on spectral pred vs reconstruction
        wdecay1= 1e-4 ## 1e-6 
        wloss1=2
        beta1=0.05
        ##
        nb_filters_enc = 32
        nb_filters_dec = 32
        dropout = 0.3 ## 0.3
        conv1_channels = 16
        conv2_channels = 32
        wdecay2= 1e-4 ##1e-6 
        wloss2=1 
        beta2=0.05 ## 0.18
        ##
        log_message(f'total_samples, batch_size, latent_dim, num_sample_exp, exploration_steps: {total_samples}, {batch_size}, {latent_dim}, {num_sample_exp}, {exploration_steps}')
        print(f'total_samples, batch_size, latent_dim, num_sample_exp, exploration_steps: {total_samples}, {batch_size}, {latent_dim}, {num_sample_exp}, {exploration_steps}')
        
        idx1, idx2 = 1568, 1634
        for i, (cy, cx) in enumerate(coordinates):
            if cy==35 and cx == 19:
                idx1 = i
            if cy==36 and cx == 38:
                idx2 = i
        print(idx1, idx2)
        
        ##
        ## plot to show patch in full image
        ##
        #
        print(X.shape, y.shape, coordinates.shape, vstep.shape)
        
        idx = idx1 ## 956
        spectra, patch = X[idx], y[idx]  # shape: (16, 16)
        coord_y, coord_x = coordinates[idx]  # coordinates are in (y, x) format
        print(coordinates[idx])
        # Plot
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 4, 1)
        plt.plot(spectra)  # mark the patch center
        plt.title("")
        plt.xlabel("Dimension")
        plt.ylabel("Intensity")
        
        plt.subplot(1, 4, 2)
        plt.plot(v_step, spectra)  # mark the patch center
        plt.title("")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Intensity")
        
        plt.subplot(1, 4, 3)
        plt.imshow(full_image, origin='lower', cmap='viridis')
        plt.scatter(coord_x, coord_y, c='white', s=100, marker='X')  # mark the patch center
        plt.title("Patch Location on Full Image")
        
        plt.subplot(1, 4, 4)
        plt.imshow(patch, cmap='viridis', origin='lower')
        plt.title("Extracted Patch")
        
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{exp_no}_patch_no_{idx}.png", dpi=300)  # <- Save here
        plt.show()
        
        ##
        ## plot to show patch in full image
        ##
        #
        print(X.shape, y.shape, coordinates.shape, vstep.shape)
        
        idx = idx2 ## 1010
        spectra, patch = X[idx], y[idx]  # shape: (16, 16)
        coord_y, coord_x = coordinates[idx]  # coordinates are in (y, x) format
        print(coordinates[idx])
        # Plot
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 4, 1)
        plt.plot(spectra)  # mark the patch center
        plt.title("")
        plt.xlabel("Dimension")
        plt.ylabel("Intensity")
        
        plt.subplot(1, 4, 2)
        plt.plot(v_step, spectra)  # mark the patch center
        plt.title("")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Intensity")
        
        plt.subplot(1, 4, 3)
        plt.imshow(full_image, origin='lower', cmap='viridis')
        plt.scatter(coord_x, coord_y, c='white', s=100, marker='X')  # mark the patch center
        plt.title("Patch Location on Full Image")
        
        plt.subplot(1, 4, 4)
        plt.imshow(patch, cmap='viridis', origin='lower')
        plt.title("Extracted Patch")
        
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{exp_no}_patch_no_{idx}.png", dpi=300)  # <- Save here
        plt.show()
        
        
        def get_noisy_spectra(coordinates, X, noisy_map, r2_score):
            has_seen = False
            coords = np.array(coordinates)
            XN = X.copy()
            noise_level = np.zeros(len(X))
            r2_level = np.zeros(len(X))    
            noisy_indices = []
            for i, (cy, cx) in enumerate(coords):
                noise_val = noisy_map[int(cy), int(cx)]
                noise_level[i] = noise_val
                r2_val = r2_score[int(cy), int(cx)]                
                r2_level[i] = r2_val
                if noise_val > 0:
                    noisy_indices.append(i)
                    if not has_seen:
                        has_seen = True
                        print(i, cy, cx)
            noisy_indices = np.array(noisy_indices)
            return coords, XN, noise_level, noisy_indices, r2_level
            
        coords, XN, noise_level, noisy_indices, r2_level = get_noisy_spectra(coordinates, X, noisy_map, r2_score)
        
        print(coords.shape, X.shape, XN.shape, noise_level.shape, noisy_indices.shape, r2_level.shape)
        
        ##
        ## plot to show noise induction
        ##
        plt.figure(figsize=(4, 4))
        plt.imshow(full_image, origin='lower', cmap='viridis')
        # plt.scatter(coords[noisy_indices, 1], coords[noisy_indices, 0], s=25, c='none', edgecolors='red', marker='o', linewidths=0.8, label='Noisy pixels')
        sc = plt.scatter(coords[:, 1], coords[:, 0], c=noise_level, cmap='hot', s=10)
        plt.colorbar(sc, label="Noise Scale")
        plt.title("Noise Map")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{exp_no}_noise_map_true.png", dpi=300)  # <- Save here
        plt.show()
        
        ##
        ## plot to show r2 score
        ##
        plt.figure(figsize=(4, 4))
        plt.imshow(full_image, origin='lower', cmap='viridis')
        sc = plt.scatter(coords[:, 1], coords[:, 0], c=r2_level, cmap='hot', s=10)
        plt.colorbar(sc, label="R2-score Scale")
        plt.title("R2-score Map")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{exp_no}_r2_map_true.png", dpi=300)  # <- Save here
        plt.show()
        
        ##
        ## plot showing noiseless vs noisy
        ##
        plt.figure(figsize=(4, 3))
        j = idx1 ## 956
        for i in range(j, j+1):  # Plot first 10 spectra
            plt.plot(X[i], label=f"R2-score: {r2_level[j]:.2f}", alpha=0.7, color='b')    
        plt.title("")
        plt.xlabel("Dimension")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{exp_no}_clean_vs_noisy_{j}.png", dpi=300)  # <- Save here
        plt.show()
        
        ##
        ## plot showing noiseless vs noisy
        ##
        plt.figure(figsize=(4, 3))
        j = idx2 ## 1010 # 1220
        for i in range(j, j+1):  # Plot first 10 spectra 
            plt.plot(XN[i], label=f"R2-score: {r2_level[j]:.2f}", alpha=0.7, color= 'r')
        plt.title("")
        plt.xlabel("Dimension")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{exp_no}_clean_vs_noisy_{j}.png", dpi=300)  # <- Save here
        plt.show()
        
        # np.argmax(r2_level)# Get sorted indices
        # sorted_idx = np.argsort(r2_level)
        
        # # Bottom 10 (lowest R²)
        # min_10_idx = sorted_idx[350:360]
        
        # # Top 10 (highest R²)
        # max_10_idx = sorted_idx[500:600]
        
        # print("Lowest 10 indices:", min_10_idx)
        # print("Highest 10 indices:", max_10_idx)
        
        ##
        ## plot showing noiseless vs noisy (sample j=720)
        ##
        plt.figure(figsize=(4, 3))
        j = idx2 ## 1010
        plt.plot(v_step, XN[j], label=f"R2-score: {r2_level[j]:.2f}", alpha=0.7, color='r')
        plt.title(f"Sample {j}: Clean vs Noisy")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{exp_no}_clean_vs_noisy_voltage_{j}.png", dpi=300)
        plt.show()
        
        ##
        ## plot showing noiseless vs noisy (sample j=705)
        ##
        v_step = np.array(v_step)  # Ensure it's a NumPy array of shape (256,)
        
        plt.figure(figsize=(4, 3))
        j = idx1 ## 956
        plt.plot(v_step, X[j], label=f"R2-score: {r2_level[j]:.2f}", alpha=0.7, color='b')
        plt.title(f"Sample {j}: Clean vs Noisy")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{exp_no}_clean_vs_noisy_voltage_{j}.png", dpi=300)
        plt.show()
        
        X_train_val, X_test, y_train_val, y_test, indices_train_val, indices_test = train_test_split(
            X, y, indices, test_size=r_test, random_state=seed
        )
        X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(
            X_train_val, y_train_val, indices_train_val, test_size=r_val, random_state=seed
        )
        XN_train, XN_val, XN_test = XN[indices_train], XN[indices_val], XN[indices_test]
        log_message(f'Train: {X_train.shape}, {y_train.shape}')
        log_message(f'Validation: {X_val.shape}, {y_val.shape}')
        log_message(f'Test: {X_test.shape}, {y_test.shape}')
        log_message(f'len(indices_train), len(indices_val), len(indices_test): {len(indices_train)}, {len(indices_val)}, {len(indices_test)}') 
        log_message(f'indices_train[0], indices_val[0], indices_test[0]: {indices_train[0]}, {indices_val[0]}, {indices_test[0]}')    
        try:
            plot_image_distribution(y_train, y_val, y_test, exp_no=exp_no, latent_dim=latent_dim, alpha=alpha, beta=beta, gamma=gamma, r_main=r_main, step=-1)
        except Exception as e:
            log_message(f'Error ({exp_no}, {latent_dim}, {alpha}, {beta}, {gamma}, {r_main}, {i}): {e}')
        print(X_train.shape, X_val.shape, X_test.shape)
        print(XN_train.shape, XN_val.shape, XN_test.shape)
        print(y_train.shape, y_val.shape, y_test.shape)
        
        #######################################
        ## random
        #######################################
        
        r2_scores = []
        for idx in indices_train: ## indices_train_val
            coord = coordinates[idx]
            avg_r2 = r2_level[idx]    
            r2_scores.append((coord, avg_r2))
        print('len(r2_scores): ', len(r2_scores))
        
        coords_xy = np.array([[c[1], c[0]] for c, _ in r2_scores])  
        r2_vals = np.array([score for _, score in r2_scores])    
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
        # gp.fit(coords_xy, r2_vals)
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        gp_prev = copy.deepcopy(gp)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                gp.fit(coords_xy, r2_vals)
                if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                    print("⚠️ GP fit issued ConvergenceWarning — reverting to previous model.")
                    gp = copy.deepcopy(gp_prev)
                else:
                    print("✅ GP fit succeeded without warnings.")
        except Exception as e:
            print(f"⚠️ GP fit failed with error: {e}")
            gp = copy.deepcopy(gp_prev)

        gpA = copy.deepcopy(gp)
        gpB = copy.deepcopy(gp)
        gpC = copy.deepcopy(gp)
        gpD = copy.deepcopy(gp)
        gpE = copy.deepcopy(gp)
        
        # ##
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        # im1 = axs[0].imshow(full_image, origin='lower', cmap='viridis')
        sc1 = axs[0].scatter(coords_xy[:, 0], coords_xy[:, 1], c=r2_vals, cmap='hot', s=10, norm=Normalize(vmin=r2_vals.min(), vmax=r2_vals.max()))
        axs[0].set_title("Train Data for GatedGP")
        axs[0].axis("off")
        fig.colorbar(sc1, ax=axs[0], fraction=0.046, pad=0.04, label="R2-score")
        ##
        coords_all_xy = np.array([[c[1], c[0]] for c in coordinates])
        r2_pred_all, std_pred_all = gpA.predict(coords_all_xy, return_std=True)
        r2_pred_all = norm_0to1(r2_pred_all)
        ##
        r2_pred_all_thresh = np.copy(r2_pred_all)
        r2_pred_all_thresh[r2_pred_all_thresh < r2_thresholdA] = np.min(r2_pred_all)
        r2_pred_all = r2_pred_all_thresh
        ##
        im2 = axs[1].imshow(full_image, origin='lower', cmap='viridis')
        sc2 = axs[1].scatter( coords_all_xy[:, 0], coords_all_xy[:, 1], c=r2_pred_all, cmap='hot', s=10, norm=Normalize(vmin=r2_pred_all.min(), vmax=r2_pred_all.max()) )
        axs[1].set_title("Prediction by GatedGP")
        axs[1].axis("off")
        fig.colorbar(sc2, ax=axs[1], fraction=0.046, pad=0.04, label="R2-score")
        ##
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{exp_no}_noise_map_GP_-1a.png", dpi=300)
        plt.show()
        
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        
        # --- Left: Train Data (with full image background) ---
        axs[0].imshow(full_image, origin='lower', cmap='viridis')  # <- add background image
        sc1 = axs[0].scatter(
            coords_xy[:, 0], coords_xy[:, 1],
            c=r2_vals, cmap='hot', s=10,
            norm=Normalize(vmin=r2_vals.min(), vmax=r2_vals.max()),
            edgecolors='k', linewidths=0.3
        )
        axs[0].set_title("Train Data for GatedGP")
        axs[0].axis("off")
        fig.colorbar(sc1, ax=axs[0], fraction=0.046, pad=0.04, label="R²-score")
        
        # --- Right: Prediction ---
        axs[1].imshow(full_image, origin='lower', cmap='viridis')
        sc2 = axs[1].scatter(
            coords_all_xy[:, 0], coords_all_xy[:, 1],
            c=r2_pred_all, cmap='hot', s=10,
            norm=Normalize(vmin=r2_pred_all.min(), vmax=r2_pred_all.max()),
            edgecolors='k', linewidths=0.3
        )
        axs[1].set_title("Prediction by GatedGP")
        axs[1].axis("off")
        fig.colorbar(sc2, ax=axs[1], fraction=0.046, pad=0.04, label="R²-score")
        
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{exp_no}_noise_map_GP_-1b.png", dpi=300)
        plt.show()
        
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(device)
        
        train_dataset_init = ImageDataset(norm_0to1_axis1(XN_train), y_train)
        val_dataset_init = ImageDataset(norm_0to1_axis1(XN_val), y_val)
        spec2im_init = spec2im(
            feature_size=dim_in, target_size=dim_out, latent_dim=latent_dim, 
            nb_filters_enc=nb_filters_enc, nb_filters_dec=nb_filters_dec, dropout=dropout
        ).to(device)
        spec2multi_init = spec2multi(
                feature_size=dim_in, target_size=dim_out, latent_dim=latent_dim,
                nb_filters_enc=nb_filters_enc, nb_filters_dec=nb_filters_dec, dropout=dropout
        ).to(device)
        
        spec2multi_init.enc_conv.load_state_dict(spec2im_init.enc_conv.state_dict())
        spec2multi_init.enc_fc.load_state_dict(spec2im_init.enc_fc.state_dict())
        ##
        spec2multi_init.img_fc.load_state_dict(spec2im_init.dec_fc.state_dict())
        spec2multi_init.img_conv_1.load_state_dict(spec2im_init.dec_conv_1.state_dict())
        spec2multi_init.img_conv_2.load_state_dict(spec2im_init.dec_conv_2.state_dict())
        spec2multi_init.img_atrous.load_state_dict(spec2im_init.dec_atrous.state_dict())
        spec2multi_init.img_conv_3.load_state_dict(spec2im_init.dec_conv_3.state_dict())
        spec2multi_init.img_out.load_state_dict(spec2im_init.dec_out.state_dict())
        
        error_model_init = ErrorModel(latent_dim=latent_dim, conv1_channels=conv1_channels, conv2_channels=conv2_channels).to(device)
        
        ##
        X_train_active, y_train_active = XN_train.copy(), y_train.copy()
        X_train_random, y_train_random = XN_train.copy(), y_train.copy()
        X_train_active2A, y_train_active2A = XN_train.copy(), y_train.copy()    
        X_train_active2B, y_train_active2B = XN_train.copy(), y_train.copy()    
        X_train_active2C, y_train_active2C = XN_train.copy(), y_train.copy()    
        X_train_active2D, y_train_active2D = XN_train.copy(), y_train.copy()    
        X_train_active2E, y_train_active2E = XN_train.copy(), y_train.copy()    
        
        X_train_active3, y_train_active3 = XN_train.copy(), y_train.copy()    
        ##
        X_val_active, y_val_active = XN_val.copy(), y_val.copy()
        X_val_random, y_val_random = XN_val.copy(), y_val.copy()
        X_val_active2A, y_val_active2A = XN_val.copy(), y_val.copy()
        X_val_active2B, y_val_active2B = XN_val.copy(), y_val.copy()
        X_val_active2C, y_val_active2C = XN_val.copy(), y_val.copy()
        X_val_active2D, y_val_active2D = XN_val.copy(), y_val.copy()
        X_val_active2E, y_val_active2E = XN_val.copy(), y_val.copy()        
        X_val_active3, y_val_active3 = XN_val.copy(), y_val.copy()
        ##
        X_test_active, XN_test_active, y_test_active = X_test.copy(), XN_test.copy(), y_test.copy()
        X_test_random, XN_test_random, y_test_random = X_test.copy(), XN_test.copy(), y_test.copy()
        X_test_active2A, XN_test_active2A, y_test_active2A = X_test.copy(), XN_test.copy(), y_test.copy()
        X_test_active2B, XN_test_active2B, y_test_active2B = X_test.copy(), XN_test.copy(), y_test.copy()
        X_test_active2C, XN_test_active2C, y_test_active2C = X_test.copy(), XN_test.copy(), y_test.copy()
        X_test_active2D, XN_test_active2D, y_test_active2D = X_test.copy(), XN_test.copy(), y_test.copy()
        X_test_active2E, XN_test_active2E, y_test_active2E = X_test.copy(), XN_test.copy(), y_test.copy()        
        X_test_active3, XN_test_active3, y_test_active3 = X_test.copy(), XN_test.copy(), y_test.copy()
        ##
        indices_train_active, indices_val_active, indices_test_active = indices_train.copy(), indices_val.copy(), indices_test.copy()
        indices_train_random, indices_val_random, indices_test_random = indices_train.copy(), indices_val.copy(), indices_test.copy()
        indices_train_active2A, indices_val_active2A, indices_test_active2A = indices_train.copy(), indices_val.copy(), indices_test.copy()
        indices_train_active2B, indices_val_active2B, indices_test_active2B = indices_train.copy(), indices_val.copy(), indices_test.copy()
        indices_train_active2C, indices_val_active2C, indices_test_active2C = indices_train.copy(), indices_val.copy(), indices_test.copy()
        indices_train_active2D, indices_val_active2D, indices_test_active2D = indices_train.copy(), indices_val.copy(), indices_test.copy()
        indices_train_active2E, indices_val_active2E, indices_test_active2E = indices_train.copy(), indices_val.copy(), indices_test.copy()
        
        indices_train_active3, indices_val_active3, indices_test_active3 = indices_train.copy(), indices_val.copy(), indices_test.copy()
        ##
        indices_train_active2A_gp = indices_train.copy()
        indices_train_active2B_gp = indices_train.copy()
        indices_train_active2C_gp = indices_train.copy()
        indices_train_active2D_gp = indices_train.copy()
        indices_train_active2E_gp = indices_train.copy()
        
        ## change
        
        ## Logs
        log_error_random = []
        log_error_active = []
        log_error_active2A = []
        log_error_active2B = []
        log_error_active2C = []
        log_error_active2D = []
        log_error_active2E = []
        
        log_error_active3 = []
        ratios_active = []
        ratios_random = []
        ratios_active2A = []
        ratios_active2B = []
        ratios_active2C = []
        ratios_active2D = []
        ratios_active2E = []
        
        ratios_active3 = []
        ##
        ##
        patience = 20                 
        relative_delta = 0.1         
        wait = 0
        best_error = required_improvement = float('inf')
        for i in range(exploration_steps):
            if terminate_check:
                if i > 0 and len(log_error_active2) > 1:
                    current_error = log_error_active2[-1]
                    if best_error == float('inf'):
                        best_error = current_error  # First time assigning
                        wait = 0
                    else:
                        required_improvement = best_error * relative_delta
                        if best_error - current_error > required_improvement:
                            best_error = current_error
                            wait = 0
                        else:
                            wait += 1
                        if wait >= patience and i>=25:
                            print(f"Early stopping at step {i} (no relative improvement > {relative_delta*100:.1f}% in {patience} steps)")
                            break
                    print('patience, relative_delta, wait, best_error, current_error, required_improvement: ', patience, relative_delta, wait, best_error, current_error, required_improvement)
            else:
                pass
            print()
            print()
            print(i)
            print(len(indices_train_active), len(np.intersect1d(noisy_indices, indices_train_active)), len(indices_test_active), len(np.intersect1d(noisy_indices, indices_test_active)))
            print(len(indices_train_random), len(np.intersect1d(noisy_indices, indices_train_random)), len(indices_test_random), len(np.intersect1d(noisy_indices, indices_test_random)))    
            print(len(indices_train_active2A), len(np.intersect1d(noisy_indices, indices_train_active2A)), len(indices_test_active2A), len(np.intersect1d(noisy_indices, indices_test_active2A)))
            print(len(indices_train_active2B), len(np.intersect1d(noisy_indices, indices_train_active2B)), len(indices_test_active2B), len(np.intersect1d(noisy_indices, indices_test_active2B)))
            print(len(indices_train_active2C), len(np.intersect1d(noisy_indices, indices_train_active2C)), len(indices_test_active2C), len(np.intersect1d(noisy_indices, indices_test_active2C)))
            print(len(indices_train_active2D), len(np.intersect1d(noisy_indices, indices_train_active2D)), len(indices_test_active2D), len(np.intersect1d(noisy_indices, indices_test_active2D)))
            print(len(indices_train_active2E), len(np.intersect1d(noisy_indices, indices_train_active2E)), len(indices_test_active2E), len(np.intersect1d(noisy_indices, indices_test_active2E)))
            
            print(len(indices_train_active3), len(np.intersect1d(noisy_indices, indices_train_active3)), len(indices_test_active3), len(np.intersect1d(noisy_indices, indices_test_active3)))    
            print(len(indices_train_active2A_gp), len(np.intersect1d(noisy_indices, indices_train_active2A_gp)))        
            print(len(indices_train_active2B_gp), len(np.intersect1d(noisy_indices, indices_train_active2B_gp)))        
            print(len(indices_train_active2C_gp), len(np.intersect1d(noisy_indices, indices_train_active2C_gp)))        
            print(len(indices_train_active2D_gp), len(np.intersect1d(noisy_indices, indices_train_active2D_gp)))        
            print(len(indices_train_active2E_gp), len(np.intersect1d(noisy_indices, indices_train_active2E_gp)))        
            
            ##################################
            ## plot: noisy ratio
            ratio_active = len(np.intersect1d(noisy_indices, indices_train_active)) / len(indices_train_active)
            ratios_active.append(ratio_active)    
            ratio_random = len(np.intersect1d(noisy_indices, indices_train_random)) / len(indices_train_random)
            ratios_random.append(ratio_random)
            ratio_active2A = len(np.intersect1d(noisy_indices, indices_train_active2A)) / len(indices_train_active2A)
            ratios_active2A.append(ratio_active2A)
            ratio_active2B = len(np.intersect1d(noisy_indices, indices_train_active2B)) / len(indices_train_active2B)
            ratios_active2B.append(ratio_active2B)
            ratio_active2C = len(np.intersect1d(noisy_indices, indices_train_active2C)) / len(indices_train_active2C)
            ratios_active2C.append(ratio_active2C)
            ratio_active2D = len(np.intersect1d(noisy_indices, indices_train_active2D)) / len(indices_train_active2D)
            ratios_active2D.append(ratio_active2D)
            ratio_active2E = len(np.intersect1d(noisy_indices, indices_train_active2E)) / len(indices_train_active2E)
            ratios_active2E.append(ratio_active2E)
            
            ratio_active3 = len(np.intersect1d(noisy_indices, indices_train_active3)) / len(indices_train_active3)
            ratios_active3.append(ratio_active3)  
            ##
            plt.figure(figsize=(5, 3))
            plt.plot(ratios_active, label='Active', marker='o', markersize=4, linewidth=1.5, color='blue')
            plt.plot(ratios_random, label='Random', marker='x', markersize=4, linewidth=1.5, color='orange')
            # plt.plot(ratios_active2, label='ActiveQC', marker='+', markersize=4, linewidth=1.5, color='green')
            plt.plot(ratios_active2A, label=f'ActiveQC τ={r2_thresholdA}', marker='+', markersize=4, linewidth=1.5, color='#2d8a2d')
            plt.plot(ratios_active2B, label=f'ActiveQC τ={r2_thresholdB}', marker='+', markersize=4, linewidth=1.5, color='#4CAF50')
            plt.plot(ratios_active2C, label=f'ActiveQC τ={r2_thresholdC}', marker='+', markersize=4, linewidth=1.5, color='#81C784')
            plt.plot(ratios_active2D, label=f'ActiveQC τ={r2_thresholdD}', marker='+', markersize=4, linewidth=1.5, color='#A5D6A7')
            plt.plot(ratios_active2E, label=f'ActiveQC τ={r2_thresholdE}', marker='+', markersize=4, linewidth=1.5, color='#C8E6C9')            
            plt.plot(ratios_active3, label='ActiveMT', marker='*', markersize=4, linewidth=1.5, color='purple')    
            plt.xlabel("Exploration Step")
            plt.ylabel("Ratio of Noisy Indices in Training Set")
            plt.title("")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"outputs/plots/{exp_no}_noisy_ratio_{i}.png", dpi=300)
            plt.show()
            ##################################
            ##################################
            ## plot: sample location
            coords_active = np.array([coordinates[idx] for idx in indices_train_active])
            coords_random = np.array([coordinates[idx] for idx in indices_train_random])
            coords_active2A = np.array([coordinates[idx] for idx in indices_train_active2A])
            coords_active2B = np.array([coordinates[idx] for idx in indices_train_active2B])
            coords_active2C = np.array([coordinates[idx] for idx in indices_train_active2C])
            coords_active2D = np.array([coordinates[idx] for idx in indices_train_active2D])
            coords_active2E = np.array([coordinates[idx] for idx in indices_train_active2E])

            coords_active3 = np.array([coordinates[idx] for idx in indices_train_active3])    
            coords_noisy = np.array([coordinates[idx] for idx in noisy_indices])
            # Plot all training sets over full image, and mark noisy regions
            plt.figure(figsize=(6, 6))
            plt.imshow(full_image, origin='lower', cmap='gray', alpha=0.35)  # low-opacity neutral background
            plt.scatter(coords_noisy[:, 1], coords_noisy[:, 0], c='red', s=20, label="Noisy", alpha=0.8, marker='s', edgecolors='black', linewidths=0.5)
            plt.scatter(coords_active[:, 1], coords_active[:, 0], c='blue', s=20, label="Active", alpha=0.9, marker='o')    
            plt.scatter(coords_random[:, 1], coords_random[:, 0], c='orange', s=20, label="Random", alpha=0.9, marker='x')    
            # plt.scatter(coords_active2[:, 1], coords_active2[:, 0], c='green', s=20, label="ActiveQC", alpha=0.9, marker='+') 
            plt.scatter(coords_active2A[:, 1], coords_active2A[:, 0], c='#2d8a2d', s=20, label=f"ActiveQC τ={r2_thresholdA}", alpha=0.9, marker='+')
            plt.scatter(coords_active2B[:, 1], coords_active2B[:, 0], c='#4CAF50', s=20, label=f"ActiveQC τ={r2_thresholdB}", alpha=0.9, marker='+')
            plt.scatter(coords_active2C[:, 1], coords_active2C[:, 0], c='#81C784', s=20, label=f"ActiveQC τ={r2_thresholdC}", alpha=0.9, marker='+')
            plt.scatter(coords_active2D[:, 1], coords_active2D[:, 0], c='#A5D6A7', s=20, label=f"ActiveQC τ={r2_thresholdD}", alpha=0.9, marker='+')
            plt.scatter(coords_active2E[:, 1], coords_active2E[:, 0], c='#C8E6C9', s=20, label=f"ActiveQC τ={r2_thresholdE}", alpha=0.9, marker='+')
            plt.scatter(coords_active3[:, 1], coords_active3[:, 0], c='purple', s=20, label="ActiveMT", alpha=0.9, marker='*')        
            plt.legend(loc='upper right', fontsize=10)
            plt.title("Sample Locations vs Noisy Region")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"outputs/plots/{exp_no}_sample_locations_vs_noise_{i}.png", dpi=300)
            plt.show()
            ##################################
            log_message(f'')
            log_message(f'')
            log_message(f'')
            log_message(f'Exploration step: {i}/{exploration_steps}')
            num_epochs = init_num_epochs if i == 0 else 50 ## 50 
            # lr = init_lr if i == 0 else 1e-3 ## 1e-3 ## lowering from the initial may get it stuck into the local minima
            # lr = init_lr * (0.1 ** (i / exploration_steps)) if i > 0 else init_lr ## exp decay
            lr = init_lr * (1 - i / exploration_steps) if i > 0 else init_lr ## linear decay        
            ##
            ## active
            ## train model
            if i==0:
                spec2im_active = copy.deepcopy(spec2im_init)
                ##
                mask_active_clean = ~np.isin(indices_test_active, noisy_indices)
                X_test_active_clean = X_test_active[mask_active_clean]
                y_test_active_clean = y_test_active[mask_active_clean]
                errors_test_active = compute_mse_s2i(spec2im_active, norm_0to1_axis1(X_test_active_clean), y_test_active_clean, device)
                ##
                log_error_active.append(np.mean(errors_test_active))
                
            train_dataset_active = ImageDataset(norm_0to1_axis1(X_train_active), y_train_active)
            val_dataset_active = ImageDataset(norm_0to1_axis1(X_val_active), y_val_active)
            ## change
            spec2im_active = train_spec2im(spec2im_active, train_dataset_active, val_dataset_active, device, num_epochs=num_epochs, lr=lr, batch_size=batch_size, wdecay=wdecay1, wloss=wloss1, beta=beta1)
            ## errors
            errors_train_active = compute_mse_s2i(spec2im_active, norm_0to1_axis1(X_train_active), y_train_active, device)    
            errors_val_active = compute_mse_s2i(spec2im_active, norm_0to1_axis1(X_val_active), y_val_active, device)        
            ##
            mask_active_clean = ~np.isin(indices_test_active, noisy_indices)
            X_test_active_clean = X_test_active[mask_active_clean]
            y_test_active_clean = y_test_active[mask_active_clean]
            errors_test_active = compute_mse_s2i(spec2im_active, norm_0to1_axis1(X_test_active_clean), y_test_active_clean, device)    
            # errors_test_active = compute_mse_s2i(spec2im_active, X_test_active, y_test_active, device)
            ##
            log_error_active.append(np.mean(errors_test_active))
            ##
            ## train error model
            spec2im_active.eval()
            with torch.no_grad():
                X_train_active_tensor = torch.tensor(X_train_active, dtype=torch.float32).unsqueeze(1).to(device)
                X_val_active_tensor = torch.tensor(X_val_active, dtype=torch.float32).unsqueeze(1).to(device)        
                X_test_active_tensor = torch.tensor(XN_test_active, dtype=torch.float32).unsqueeze(1).to(device)            
                X_train_active_latent = spec2im_active.encoder(X_train_active_tensor).cpu().numpy()        
                X_val_active_latent = spec2im_active.encoder(X_val_active_tensor).cpu().numpy()
                X_test_active_latent = spec2im_active.encoder(X_test_active_tensor).cpu().numpy()
                ##
                # Normalize using training set stats
                min_vals = X_train_active_latent.min(axis=0, keepdims=True)
                max_vals = X_train_active_latent.max(axis=0, keepdims=True)
                eps = 1e-8
                range_vals = max_vals - min_vals + eps
                X_train_active_latent = (X_train_active_latent - min_vals) / range_vals
                X_val_active_latent   = (X_val_active_latent - min_vals) / range_vals
                X_test_active_latent  = (X_test_active_latent - min_vals) / range_vals            
                ##
            train_error_dataset_active = ErrorDataset(X_val_active_latent, norm_0to1(errors_val_active))
            val_error_dataset_active = ErrorDataset(X_train_active_latent, norm_0to1(errors_train_active))        
            if i==0:
                error_model_active = copy.deepcopy(error_model_init) ## ErrorModel(latent_dim).to(device)
            ## change
            error_model_active = train_error_model(error_model_active, train_error_dataset_active, val_error_dataset_active, device, num_epochs=10*num_epochs, lr=0.5*lr, batch_size=batch_size, 
                                                   wdecay=wdecay2, wloss=wloss2, beta=beta2)
            ## learning/update 
            ############################
            if alpha == 0 and beta == 0 and gamma == 0:
                indices_next_active = rng.choice(len(XN_test_active), num_sample_exp, replace=False)
            else:
                predicted_errors_active = norm_0to1(predict_error_model(error_model_active, X_test_active_latent, device)) ## for alpha
                distances_active = norm_0to1(compute_distances(X_test_active_latent, X_train_active_latent)) ## for beta
                sims_active = norm_0to1(compute_sims(X_test_active_latent)) ## for gamma
                acquisition_values_active = acquisition_function(predicted_errors_active, distances_active, sims_active, alpha=alpha, beta=0, gamma=0) ## pure active (1, 0, 0)
                indices_next_active = np.argsort(acquisition_values_active)[-num_sample_exp:]
            ############################    
            X_train_active = np.vstack((X_train_active, XN_test_active[indices_next_active]))
            y_train_active = np.vstack((y_train_active, y_test_active[indices_next_active]))
            X_test_active = np.delete(X_test_active, indices_next_active, axis=0)
            y_test_active = np.delete(y_test_active, indices_next_active, axis=0)
            XN_test_active = np.delete(XN_test_active, indices_next_active, axis=0)    
            selected_indices_active = indices_test_active[indices_next_active]        
            # print('active')
            # print(len(np.intersect1d(noisy_indices, indices_train_active)), len(np.intersect1d(noisy_indices, selected_indices_active)), len(np.intersect1d(noisy_indices, indices_test_active)))                
            indices_train_active = np.concatenate((indices_train_active, selected_indices_active))
            indices_test_active = np.delete(indices_test_active, indices_next_active, axis=0)    
            # print(len(np.intersect1d(noisy_indices, indices_train_active)), len(np.intersect1d(noisy_indices, selected_indices_active)), len(np.intersect1d(noisy_indices, indices_test_active)))
            # print(len(selected_indices_active))
            # print()    
            log_message(f'active shape: {X_train_active.shape}, {y_train_active.shape}, {X_test_active.shape}, {y_test_active.shape}, {XN_test_active.shape}')
        
        
        
        
        
            ## random
            ## train model
            if i == 0:
                # spec2im_random = spec2im(feature_size=dim_in, target_size=dim_out, latent_dim=latent_dim).to(device)
                spec2im_random = copy.deepcopy(spec2im_init)
                ##
                mask_random_clean = ~np.isin(indices_test_random, noisy_indices)
                X_test_random_clean = X_test_random[mask_random_clean]
                y_test_random_clean = y_test_random[mask_random_clean]        
                errors_test_random = compute_mse_s2i(spec2im_random, norm_0to1_axis1(X_test_random_clean), y_test_random_clean, device)
                ##
                log_error_random.append(np.mean(errors_test_random))
                
            train_dataset_random = ImageDataset(norm_0to1_axis1(X_train_random), y_train_random)
            val_dataset_random = ImageDataset(norm_0to1_axis1(X_val_random), y_val_random)
            ## change
            spec2im_random = train_spec2im(spec2im_random, train_dataset_random, val_dataset_random, device, num_epochs=num_epochs, lr=lr, batch_size=batch_size, wdecay=wdecay1, wloss=wloss1, beta=beta1)
            ## errors
            mask_random_clean = ~np.isin(indices_test_random, noisy_indices)
            X_test_random_clean = X_test_random[mask_random_clean]
            y_test_random_clean = y_test_random[mask_random_clean]        
            errors_test_random = compute_mse_s2i(spec2im_random, norm_0to1_axis1(X_test_random_clean), y_test_random_clean, device)    
            log_error_random.append(np.mean(errors_test_random))
            ## learning/update
            ############################
            indices_next_random = rng.choice(len(XN_test_random), num_sample_exp, replace=False)
            ############################
            X_train_random = np.vstack((X_train_random, XN_test_random[indices_next_random]))
            y_train_random = np.vstack((y_train_random, y_test_random[indices_next_random]))    
            X_test_random = np.delete(X_test_random, indices_next_random, axis=0)
            y_test_random = np.delete(y_test_random, indices_next_random, axis=0)
            XN_test_random = np.delete(XN_test_random, indices_next_random, axis=0)    
            selected_indices_random = indices_test_random[indices_next_random]      
            # print('random')
            # print(len(np.intersect1d(noisy_indices, indices_train_random)), len(np.intersect1d(noisy_indices, selected_indices_random)), len(np.intersect1d(noisy_indices, indices_test_random)))            
            indices_train_random = np.concatenate((indices_train_random, selected_indices_random))
            indices_test_random = np.delete(indices_test_random, indices_next_random, axis=0)   
            # print(len(np.intersect1d(noisy_indices, indices_train_random)), len(np.intersect1d(noisy_indices, selected_indices_random)), len(np.intersect1d(noisy_indices, indices_test_random)))
            # print(len(selected_indices_random))
            # print()    
            log_message(f'random shape: {X_train_random.shape}, {y_train_random.shape}, {X_test_random.shape}, {y_test_random.shape}, {XN_test_random.shape}')
        
        
            ## logging, plotting, checkpoints
            ## active2A
            ## train model
            if i==0:
                spec2im_active2A = copy.deepcopy(spec2im_init)
                mask_active2A_clean = ~np.isin(indices_test_active2A, noisy_indices)
                X_test_active2A_clean = X_test_active2A[mask_active2A_clean]
                y_test_active2A_clean = y_test_active2A[mask_active2A_clean]        
                errors_test_active2A = compute_mse_s2i(spec2im_active2A, norm_0to1_axis1(X_test_active2A_clean), y_test_active2A_clean, device)
                log_error_active2A.append(np.mean(errors_test_active2A))
                
            train_dataset_active2A = ImageDataset(norm_0to1_axis1(X_train_active2A), y_train_active2A)
            val_dataset_active2A = ImageDataset(norm_0to1_axis1(X_val_active2A), y_val_active2A)
            ## change
            spec2im_active2A = train_spec2im(spec2im_active2A, train_dataset_active2A, val_dataset_active2A, device, num_epochs=num_epochs, lr=lr, batch_size=batch_size, wdecay=wdecay1, wloss=wloss1, beta=beta1)
            ## errors
            errors_train_active2A = compute_mse_s2i(spec2im_active2A, norm_0to1_axis1(X_train_active2A), y_train_active2A, device)    
            errors_val_active2A = compute_mse_s2i(spec2im_active2A, norm_0to1_axis1(X_val_active2A), y_val_active2A, device)        
        
            mask_active2A_clean = ~np.isin(indices_test_active2A, noisy_indices)
            X_test_active2A_clean = X_test_active2A[mask_active2A_clean]
            y_test_active2A_clean = y_test_active2A[mask_active2A_clean]        
            errors_test_active2A = compute_mse_s2i(spec2im_active2A, norm_0to1_axis1(X_test_active2A_clean), y_test_active2A_clean, device)    
            log_error_active2A.append(np.mean(errors_test_active2A))
            ##
            ## train error model
            spec2im_active2A.eval()
            with torch.no_grad():
                X_train_active2A_tensor = torch.tensor(X_train_active2A, dtype=torch.float32).unsqueeze(1).to(device)
                X_val_active2A_tensor = torch.tensor(X_val_active2A, dtype=torch.float32).unsqueeze(1).to(device)        
                X_test_active2A_tensor = torch.tensor(XN_test_active2A, dtype=torch.float32).unsqueeze(1).to(device)            
                X_train_active2A_latent = spec2im_active2A.encoder(X_train_active2A_tensor).cpu().numpy()        
                X_val_active2A_latent = spec2im_active2A.encoder(X_val_active2A_tensor).cpu().numpy()
                X_test_active2A_latent = spec2im_active2A.encoder(X_test_active2A_tensor).cpu().numpy()
                ##
                # Normalize using training set stats
                min_vals = X_train_active2A_latent.min(axis=0, keepdims=True)
                max_vals = X_train_active2A_latent.max(axis=0, keepdims=True)
                eps = 1e-8
                range_vals = max_vals - min_vals + eps
                X_train_active2A_latent = (X_train_active2A_latent - min_vals) / range_vals
                X_val_active2A_latent   = (X_val_active2A_latent - min_vals) / range_vals
                X_test_active2A_latent  = (X_test_active2A_latent - min_vals) / range_vals            
                ##
            train_error_dataset_active2A = ErrorDataset(X_val_active2A_latent, norm_0to1(errors_val_active2A))
            val_error_dataset_active2A = ErrorDataset(X_train_active2A_latent, norm_0to1(errors_train_active2A))        
            if i==0:
                error_model_active2A = copy.deepcopy(error_model_init) ## ErrorModel(latent_dim).to(device)
            ## change
            error_model_active2A = train_error_model(error_model_active2A, train_error_dataset_active2A, val_error_dataset_active2A, device, num_epochs=10*num_epochs, lr=0.5*lr, batch_size=batch_size, 
                                                   wdecay=wdecay2, wloss=wloss2, beta=beta2)
            ## learning/update 
            ############################
            num_sample_exp2 = len(indices_train_active) - len(indices_train_active2A)
            print('===')
            print(len(indices_train_active2A), len(indices_train_active), num_sample_exp2)
            if alpha == 0 and beta == 0 and gamma == 0:
                indices_next_active2A = rng.choice(len(XN_test_active2A), num_sample_exp2, replace=False)
            else:
                predicted_errors_active2A = norm_0to1(predict_error_model(error_model_active2A, X_test_active2A_latent, device)) ## for alpha
                distances_active2A = norm_0to1(compute_distances(X_test_active2A_latent, X_train_active2A_latent)) ## for beta
                sims_active2A = norm_0to1(compute_sims(X_test_active2A_latent)) ## for gamma
                acquisition_values_active2A = acquisition_function(predicted_errors_active2A, distances_active2A, sims_active2A, alpha=alpha, beta=0, gamma=0, gp=gpA, indices=indices_test_active2A, coordinates=coordinates, r2_threshold=r2_thresholdA)
                indices_next_active2A = np.argsort(acquisition_values_active2A)[-num_sample_exp2:]
            ############################    
            X_train_active2A = np.vstack((X_train_active2A, XN_test_active2A[indices_next_active2A]))
            y_train_active2A = np.vstack((y_train_active2A, y_test_active2A[indices_next_active2A]))    
            X_test_active2A = np.delete(X_test_active2A, indices_next_active2A, axis=0)
            y_test_active2A = np.delete(y_test_active2A, indices_next_active2A, axis=0)
            XN_test_active2A = np.delete(XN_test_active2A, indices_next_active2A, axis=0)    
            selected_indices_active2A = indices_test_active2A[indices_next_active2A]        
            # print('active2A')
                  
            indices_train_active2A = np.concatenate((indices_train_active2A, selected_indices_active2A))
            indices_train_active2A_gp = np.concatenate((indices_train_active2A_gp, selected_indices_active2A))    
            indices_test_active2A = np.delete(indices_test_active2A, indices_next_active2A, axis=0)   
            # print(len(selected_indices_active2A))
            # print()        
            log_message(f'active2A shape: {X_train_active2A.shape}, {y_train_active2A.shape}, {X_test_active2A.shape}, {y_test_active2A.shape}, {XN_test_active2A.shape}')

            ## logging, plotting, checkpoints
            ## active2B
            ## train model
            if i==0:
                spec2im_active2B = copy.deepcopy(spec2im_init)
                mask_active2B_clean = ~np.isin(indices_test_active2B, noisy_indices)
                X_test_active2B_clean = X_test_active2B[mask_active2B_clean]
                y_test_active2B_clean = y_test_active2B[mask_active2B_clean]        
                errors_test_active2B = compute_mse_s2i(spec2im_active2B, norm_0to1_axis1(X_test_active2B_clean), y_test_active2B_clean, device)
                log_error_active2B.append(np.mean(errors_test_active2B))
                
            train_dataset_active2B = ImageDataset(norm_0to1_axis1(X_train_active2B), y_train_active2B)
            val_dataset_active2B = ImageDataset(norm_0to1_axis1(X_val_active2B), y_val_active2B)
            ## change
            spec2im_active2B = train_spec2im(spec2im_active2B, train_dataset_active2B, val_dataset_active2B, device, num_epochs=num_epochs, lr=lr, batch_size=batch_size, wdecay=wdecay1, wloss=wloss1, beta=beta1)
            ## errors
            errors_train_active2B = compute_mse_s2i(spec2im_active2B, norm_0to1_axis1(X_train_active2B), y_train_active2B, device)    
            errors_val_active2B = compute_mse_s2i(spec2im_active2B, norm_0to1_axis1(X_val_active2B), y_val_active2B, device)        
        
            mask_active2B_clean = ~np.isin(indices_test_active2B, noisy_indices)
            X_test_active2B_clean = X_test_active2B[mask_active2B_clean]
            y_test_active2B_clean = y_test_active2B[mask_active2B_clean]        
            errors_test_active2B = compute_mse_s2i(spec2im_active2B, norm_0to1_axis1(X_test_active2B_clean), y_test_active2B_clean, device)    
            log_error_active2B.append(np.mean(errors_test_active2B))
            ##
            ## train error model
            spec2im_active2B.eval()
            with torch.no_grad():
                X_train_active2B_tensor = torch.tensor(X_train_active2B, dtype=torch.float32).unsqueeze(1).to(device)
                X_val_active2B_tensor = torch.tensor(X_val_active2B, dtype=torch.float32).unsqueeze(1).to(device)        
                X_test_active2B_tensor = torch.tensor(XN_test_active2B, dtype=torch.float32).unsqueeze(1).to(device)            
                X_train_active2B_latent = spec2im_active2B.encoder(X_train_active2B_tensor).cpu().numpy()        
                X_val_active2B_latent = spec2im_active2B.encoder(X_val_active2B_tensor).cpu().numpy()
                X_test_active2B_latent = spec2im_active2B.encoder(X_test_active2B_tensor).cpu().numpy()
                ##
                # Normalize using training set stats
                min_vals = X_train_active2B_latent.min(axis=0, keepdims=True)
                max_vals = X_train_active2B_latent.max(axis=0, keepdims=True)
                eps = 1e-8
                range_vals = max_vals - min_vals + eps
                X_train_active2B_latent = (X_train_active2B_latent - min_vals) / range_vals
                X_val_active2B_latent   = (X_val_active2B_latent - min_vals) / range_vals
                X_test_active2B_latent  = (X_test_active2B_latent - min_vals) / range_vals            
                ##
            train_error_dataset_active2B = ErrorDataset(X_val_active2B_latent, norm_0to1(errors_val_active2B))
            val_error_dataset_active2B = ErrorDataset(X_train_active2B_latent, norm_0to1(errors_train_active2B))        
            if i==0:
                error_model_active2B = copy.deepcopy(error_model_init) ## ErrorModel(latent_dim).to(device)
            ## change
            error_model_active2B = train_error_model(error_model_active2B, train_error_dataset_active2B, val_error_dataset_active2B, device, num_epochs=10*num_epochs, lr=0.5*lr, batch_size=batch_size, 
                                                   wdecay=wdecay2, wloss=wloss2, beta=beta2)
            ## learning/update 
            ############################
            num_sample_exp2 = len(indices_train_active) - len(indices_train_active2B)
            print('===')
            print(len(indices_train_active2B), len(indices_train_active), num_sample_exp2)
            if alpha == 0 and beta == 0 and gamma == 0:
                indices_next_active2B = rng.choice(len(XN_test_active2B), num_sample_exp2, replace=False)
            else:
                predicted_errors_active2B = norm_0to1(predict_error_model(error_model_active2B, X_test_active2B_latent, device)) ## for alpha
                distances_active2B = norm_0to1(compute_distances(X_test_active2B_latent, X_train_active2B_latent)) ## for beta
                sims_active2B = norm_0to1(compute_sims(X_test_active2B_latent)) ## for gamma
                acquisition_values_active2B = acquisition_function(predicted_errors_active2B, distances_active2B, sims_active2B, alpha=alpha, beta=0, gamma=0, gp=gpB, indices=indices_test_active2B, coordinates=coordinates, r2_threshold=r2_thresholdB)
                indices_next_active2B = np.argsort(acquisition_values_active2B)[-num_sample_exp2:]
            ############################    
            X_train_active2B = np.vstack((X_train_active2B, XN_test_active2B[indices_next_active2B]))
            y_train_active2B = np.vstack((y_train_active2B, y_test_active2B[indices_next_active2B]))    
            X_test_active2B = np.delete(X_test_active2B, indices_next_active2B, axis=0)
            y_test_active2B = np.delete(y_test_active2B, indices_next_active2B, axis=0)
            XN_test_active2B = np.delete(XN_test_active2B, indices_next_active2B, axis=0)    
            selected_indices_active2B = indices_test_active2B[indices_next_active2B]        
            # print('active2B')
                  
            indices_train_active2B = np.concatenate((indices_train_active2B, selected_indices_active2B))
            indices_train_active2B_gp = np.concatenate((indices_train_active2B_gp, selected_indices_active2B))    
            indices_test_active2B = np.delete(indices_test_active2B, indices_next_active2B, axis=0)   
            # print(len(selected_indices_active2B))
            # print()        
            log_message(f'active2B shape: {X_train_active2B.shape}, {y_train_active2B.shape}, {X_test_active2B.shape}, {y_test_active2B.shape}, {XN_test_active2B.shape}')
        
            ## logging, plotting, checkpoints
            ## active2C
            ## train model
            if i==0:
                spec2im_active2C = copy.deepcopy(spec2im_init)
                mask_active2C_clean = ~np.isin(indices_test_active2C, noisy_indices)
                X_test_active2C_clean = X_test_active2C[mask_active2C_clean]
                y_test_active2C_clean = y_test_active2C[mask_active2C_clean]        
                errors_test_active2C = compute_mse_s2i(spec2im_active2C, norm_0to1_axis1(X_test_active2C_clean), y_test_active2C_clean, device)
                log_error_active2C.append(np.mean(errors_test_active2C))
                
            train_dataset_active2C = ImageDataset(norm_0to1_axis1(X_train_active2C), y_train_active2C)
            val_dataset_active2C = ImageDataset(norm_0to1_axis1(X_val_active2C), y_val_active2C)
            ## change
            spec2im_active2C = train_spec2im(spec2im_active2C, train_dataset_active2C, val_dataset_active2C, device, num_epochs=num_epochs, lr=lr, batch_size=batch_size, wdecay=wdecay1, wloss=wloss1, beta=beta1)
            ## errors
            errors_train_active2C = compute_mse_s2i(spec2im_active2C, norm_0to1_axis1(X_train_active2C), y_train_active2C, device)    
            errors_val_active2C = compute_mse_s2i(spec2im_active2C, norm_0to1_axis1(X_val_active2C), y_val_active2C, device)        
        
            mask_active2C_clean = ~np.isin(indices_test_active2C, noisy_indices)
            X_test_active2C_clean = X_test_active2C[mask_active2C_clean]
            y_test_active2C_clean = y_test_active2C[mask_active2C_clean]        
            errors_test_active2C = compute_mse_s2i(spec2im_active2C, norm_0to1_axis1(X_test_active2C_clean), y_test_active2C_clean, device)    
            log_error_active2C.append(np.mean(errors_test_active2C))
            ##
            ## train error model
            spec2im_active2C.eval()
            with torch.no_grad():
                X_train_active2C_tensor = torch.tensor(X_train_active2C, dtype=torch.float32).unsqueeze(1).to(device)
                X_val_active2C_tensor = torch.tensor(X_val_active2C, dtype=torch.float32).unsqueeze(1).to(device)        
                X_test_active2C_tensor = torch.tensor(XN_test_active2C, dtype=torch.float32).unsqueeze(1).to(device)            
                X_train_active2C_latent = spec2im_active2C.encoder(X_train_active2C_tensor).cpu().numpy()        
                X_val_active2C_latent = spec2im_active2C.encoder(X_val_active2C_tensor).cpu().numpy()
                X_test_active2C_latent = spec2im_active2C.encoder(X_test_active2C_tensor).cpu().numpy()
                ##
                # Normalize using training set stats
                min_vals = X_train_active2C_latent.min(axis=0, keepdims=True)
                max_vals = X_train_active2C_latent.max(axis=0, keepdims=True)
                eps = 1e-8
                range_vals = max_vals - min_vals + eps
                X_train_active2C_latent = (X_train_active2C_latent - min_vals) / range_vals
                X_val_active2C_latent   = (X_val_active2C_latent - min_vals) / range_vals
                X_test_active2C_latent  = (X_test_active2C_latent - min_vals) / range_vals            
                ##
            train_error_dataset_active2C = ErrorDataset(X_val_active2C_latent, norm_0to1(errors_val_active2C))
            val_error_dataset_active2C = ErrorDataset(X_train_active2C_latent, norm_0to1(errors_train_active2C))        
            if i==0:
                error_model_active2C = copy.deepcopy(error_model_init) ## ErrorModel(latent_dim).to(device)
            ## change
            error_model_active2C = train_error_model(error_model_active2C, train_error_dataset_active2C, val_error_dataset_active2C, device, num_epochs=10*num_epochs, lr=0.5*lr, batch_size=batch_size, 
                                                   wdecay=wdecay2, wloss=wloss2, beta=beta2)
            ## learning/update 
            ############################
            num_sample_exp2 = len(indices_train_active) - len(indices_train_active2C)
            print('===')
            print(len(indices_train_active2C), len(indices_train_active), num_sample_exp2)
            if alpha == 0 and beta == 0 and gamma == 0:
                indices_next_active2C = rng.choice(len(XN_test_active2C), num_sample_exp2, replace=False)
            else:
                predicted_errors_active2C = norm_0to1(predict_error_model(error_model_active2C, X_test_active2C_latent, device)) ## for alpha
                distances_active2C = norm_0to1(compute_distances(X_test_active2C_latent, X_train_active2C_latent)) ## for beta
                sims_active2C = norm_0to1(compute_sims(X_test_active2C_latent)) ## for gamma
                acquisition_values_active2C = acquisition_function(predicted_errors_active2C, distances_active2C, sims_active2C, alpha=alpha, beta=0, gamma=0, gp=gpC, indices=indices_test_active2C, coordinates=coordinates, r2_threshold=r2_thresholdC)
                indices_next_active2C = np.argsort(acquisition_values_active2C)[-num_sample_exp2:]
            ############################    
            X_train_active2C = np.vstack((X_train_active2C, XN_test_active2C[indices_next_active2C]))
            y_train_active2C = np.vstack((y_train_active2C, y_test_active2C[indices_next_active2C]))    
            X_test_active2C = np.delete(X_test_active2C, indices_next_active2C, axis=0)
            y_test_active2C = np.delete(y_test_active2C, indices_next_active2C, axis=0)
            XN_test_active2C = np.delete(XN_test_active2C, indices_next_active2C, axis=0)    
            selected_indices_active2C = indices_test_active2C[indices_next_active2C]        
            # print('active2C')
                  
            indices_train_active2C = np.concatenate((indices_train_active2C, selected_indices_active2C))
            indices_train_active2C_gp = np.concatenate((indices_train_active2C_gp, selected_indices_active2C))    
            indices_test_active2C = np.delete(indices_test_active2C, indices_next_active2C, axis=0)   
            # print(len(selected_indices_active2C))
            # print()        
            log_message(f'active2C shape: {X_train_active2C.shape}, {y_train_active2C.shape}, {X_test_active2C.shape}, {y_test_active2C.shape}, {XN_test_active2C.shape}')
        
            ## logging, plotting, checkpoints
            ## active2D
            ## train model
            if i==0:
                spec2im_active2D = copy.deepcopy(spec2im_init)
                mask_active2D_clean = ~np.isin(indices_test_active2D, noisy_indices)
                X_test_active2D_clean = X_test_active2D[mask_active2D_clean]
                y_test_active2D_clean = y_test_active2D[mask_active2D_clean]        
                errors_test_active2D = compute_mse_s2i(spec2im_active2D, norm_0to1_axis1(X_test_active2D_clean), y_test_active2D_clean, device)
                log_error_active2D.append(np.mean(errors_test_active2D))
                
            train_dataset_active2D = ImageDataset(norm_0to1_axis1(X_train_active2D), y_train_active2D)
            val_dataset_active2D = ImageDataset(norm_0to1_axis1(X_val_active2D), y_val_active2D)
            ## change
            spec2im_active2D = train_spec2im(spec2im_active2D, train_dataset_active2D, val_dataset_active2D, device, num_epochs=num_epochs, lr=lr, batch_size=batch_size, wdecay=wdecay1, wloss=wloss1, beta=beta1)
            ## errors
            errors_train_active2D = compute_mse_s2i(spec2im_active2D, norm_0to1_axis1(X_train_active2D), y_train_active2D, device)    
            errors_val_active2D = compute_mse_s2i(spec2im_active2D, norm_0to1_axis1(X_val_active2D), y_val_active2D, device)        
        
            mask_active2D_clean = ~np.isin(indices_test_active2D, noisy_indices)
            X_test_active2D_clean = X_test_active2D[mask_active2D_clean]
            y_test_active2D_clean = y_test_active2D[mask_active2D_clean]        
            errors_test_active2D = compute_mse_s2i(spec2im_active2D, norm_0to1_axis1(X_test_active2D_clean), y_test_active2D_clean, device)    
            log_error_active2D.append(np.mean(errors_test_active2D))
            ##
            ## train error model
            spec2im_active2D.eval()
            with torch.no_grad():
                X_train_active2D_tensor = torch.tensor(X_train_active2D, dtype=torch.float32).unsqueeze(1).to(device)
                X_val_active2D_tensor = torch.tensor(X_val_active2D, dtype=torch.float32).unsqueeze(1).to(device)        
                X_test_active2D_tensor = torch.tensor(XN_test_active2D, dtype=torch.float32).unsqueeze(1).to(device)            
                X_train_active2D_latent = spec2im_active2D.encoder(X_train_active2D_tensor).cpu().numpy()        
                X_val_active2D_latent = spec2im_active2D.encoder(X_val_active2D_tensor).cpu().numpy()
                X_test_active2D_latent = spec2im_active2D.encoder(X_test_active2D_tensor).cpu().numpy()
                ##
                # Normalize using training set stats
                min_vals = X_train_active2D_latent.min(axis=0, keepdims=True)
                max_vals = X_train_active2D_latent.max(axis=0, keepdims=True)
                eps = 1e-8
                range_vals = max_vals - min_vals + eps
                X_train_active2D_latent = (X_train_active2D_latent - min_vals) / range_vals
                X_val_active2D_latent   = (X_val_active2D_latent - min_vals) / range_vals
                X_test_active2D_latent  = (X_test_active2D_latent - min_vals) / range_vals            
                ##
            train_error_dataset_active2D = ErrorDataset(X_val_active2D_latent, norm_0to1(errors_val_active2D))
            val_error_dataset_active2D = ErrorDataset(X_train_active2D_latent, norm_0to1(errors_train_active2D))        
            if i==0:
                error_model_active2D = copy.deepcopy(error_model_init) ## ErrorModel(latent_dim).to(device)
            ## change
            error_model_active2D = train_error_model(error_model_active2D, train_error_dataset_active2D, val_error_dataset_active2D, device, num_epochs=10*num_epochs, lr=0.5*lr, batch_size=batch_size, 
                                                   wdecay=wdecay2, wloss=wloss2, beta=beta2)
            ## learning/update 
            ############################
            num_sample_exp2 = len(indices_train_active) - len(indices_train_active2D)
            print('===')
            print(len(indices_train_active2D), len(indices_train_active), num_sample_exp2)
            if alpha == 0 and beta == 0 and gamma == 0:
                indices_next_active2D = rng.choice(len(XN_test_active2D), num_sample_exp2, replace=False)
            else:
                predicted_errors_active2D = norm_0to1(predict_error_model(error_model_active2D, X_test_active2D_latent, device)) ## for alpha
                distances_active2D = norm_0to1(compute_distances(X_test_active2D_latent, X_train_active2D_latent)) ## for beta
                sims_active2D = norm_0to1(compute_sims(X_test_active2D_latent)) ## for gamma
                acquisition_values_active2D = acquisition_function(predicted_errors_active2D, distances_active2D, sims_active2D, alpha=alpha, beta=0, gamma=0, gp=gpD, indices=indices_test_active2D, coordinates=coordinates, r2_threshold=r2_thresholdD)
                indices_next_active2D = np.argsort(acquisition_values_active2D)[-num_sample_exp2:]
            ############################    
            X_train_active2D = np.vstack((X_train_active2D, XN_test_active2D[indices_next_active2D]))
            y_train_active2D = np.vstack((y_train_active2D, y_test_active2D[indices_next_active2D]))    
            X_test_active2D = np.delete(X_test_active2D, indices_next_active2D, axis=0)
            y_test_active2D = np.delete(y_test_active2D, indices_next_active2D, axis=0)
            XN_test_active2D = np.delete(XN_test_active2D, indices_next_active2D, axis=0)    
            selected_indices_active2D = indices_test_active2D[indices_next_active2D]        
            # print('active2D')
                  
            indices_train_active2D = np.concatenate((indices_train_active2D, selected_indices_active2D))
            indices_train_active2D_gp = np.concatenate((indices_train_active2D_gp, selected_indices_active2D))    
            indices_test_active2D = np.delete(indices_test_active2D, indices_next_active2D, axis=0)   
            # print(len(selected_indices_active2D))
            # print()        
            log_message(f'active2D shape: {X_train_active2D.shape}, {y_train_active2D.shape}, {X_test_active2D.shape}, {y_test_active2D.shape}, {XN_test_active2D.shape}')

            ## logging, plotting, checkpoints
            ## active2E
            ## train model
            if i==0:
                spec2im_active2E = copy.deepcopy(spec2im_init)
                mask_active2E_clean = ~np.isin(indices_test_active2E, noisy_indices)
                X_test_active2E_clean = X_test_active2E[mask_active2E_clean]
                y_test_active2E_clean = y_test_active2E[mask_active2E_clean]        
                errors_test_active2E = compute_mse_s2i(spec2im_active2E, norm_0to1_axis1(X_test_active2E_clean), y_test_active2E_clean, device)
                log_error_active2E.append(np.mean(errors_test_active2E))
                
            train_dataset_active2E = ImageDataset(norm_0to1_axis1(X_train_active2E), y_train_active2E)
            val_dataset_active2E = ImageDataset(norm_0to1_axis1(X_val_active2E), y_val_active2E)
            ## change
            spec2im_active2E = train_spec2im(spec2im_active2E, train_dataset_active2E, val_dataset_active2E, device, num_epochs=num_epochs, lr=lr, batch_size=batch_size, wdecay=wdecay1, wloss=wloss1, beta=beta1)
            ## errors
            errors_train_active2E = compute_mse_s2i(spec2im_active2E, norm_0to1_axis1(X_train_active2E), y_train_active2E, device)    
            errors_val_active2E = compute_mse_s2i(spec2im_active2E, norm_0to1_axis1(X_val_active2E), y_val_active2E, device)        
        
            mask_active2E_clean = ~np.isin(indices_test_active2E, noisy_indices)
            X_test_active2E_clean = X_test_active2E[mask_active2E_clean]
            y_test_active2E_clean = y_test_active2E[mask_active2E_clean]        
            errors_test_active2E = compute_mse_s2i(spec2im_active2E, norm_0to1_axis1(X_test_active2E_clean), y_test_active2E_clean, device)    
            log_error_active2E.append(np.mean(errors_test_active2E))
            ##
            ## train error model
            spec2im_active2E.eval()
            with torch.no_grad():
                X_train_active2E_tensor = torch.tensor(X_train_active2E, dtype=torch.float32).unsqueeze(1).to(device)
                X_val_active2E_tensor = torch.tensor(X_val_active2E, dtype=torch.float32).unsqueeze(1).to(device)        
                X_test_active2E_tensor = torch.tensor(XN_test_active2E, dtype=torch.float32).unsqueeze(1).to(device)            
                X_train_active2E_latent = spec2im_active2E.encoder(X_train_active2E_tensor).cpu().numpy()        
                X_val_active2E_latent = spec2im_active2E.encoder(X_val_active2E_tensor).cpu().numpy()
                X_test_active2E_latent = spec2im_active2E.encoder(X_test_active2E_tensor).cpu().numpy()
                ##
                # Normalize using training set stats
                min_vals = X_train_active2E_latent.min(axis=0, keepdims=True)
                max_vals = X_train_active2E_latent.max(axis=0, keepdims=True)
                eps = 1e-8
                range_vals = max_vals - min_vals + eps
                X_train_active2E_latent = (X_train_active2E_latent - min_vals) / range_vals
                X_val_active2E_latent   = (X_val_active2E_latent - min_vals) / range_vals
                X_test_active2E_latent  = (X_test_active2E_latent - min_vals) / range_vals            
                ##
            train_error_dataset_active2E = ErrorDataset(X_val_active2E_latent, norm_0to1(errors_val_active2E))
            val_error_dataset_active2E = ErrorDataset(X_train_active2E_latent, norm_0to1(errors_train_active2E))        
            if i==0:
                error_model_active2E = copy.deepcopy(error_model_init) ## ErrorModel(latent_dim).to(device)
            ## change
            error_model_active2E = train_error_model(error_model_active2E, train_error_dataset_active2E, val_error_dataset_active2E, device, num_epochs=10*num_epochs, lr=0.5*lr, batch_size=batch_size, 
                                                   wdecay=wdecay2, wloss=wloss2, beta=beta2)
            ## learning/update 
            ############################
            num_sample_exp2 = len(indices_train_active) - len(indices_train_active2E)
            print('===')
            print(len(indices_train_active2E), len(indices_train_active), num_sample_exp2)
            if alpha == 0 and beta == 0 and gamma == 0:
                indices_next_active2E = rng.choice(len(XN_test_active2E), num_sample_exp2, replace=False)
            else:
                predicted_errors_active2E = norm_0to1(predict_error_model(error_model_active2E, X_test_active2E_latent, device)) ## for alpha
                distances_active2E = norm_0to1(compute_distances(X_test_active2E_latent, X_train_active2E_latent)) ## for beta
                sims_active2E = norm_0to1(compute_sims(X_test_active2E_latent)) ## for gamma
                acquisition_values_active2E = acquisition_function(predicted_errors_active2E, distances_active2E, sims_active2E, alpha=alpha, beta=0, gamma=0, gp=gpE, indices=indices_test_active2E, coordinates=coordinates, r2_threshold=r2_thresholdE)
                indices_next_active2E = np.argsort(acquisition_values_active2E)[-num_sample_exp2:]
            ############################    
            X_train_active2E = np.vstack((X_train_active2E, XN_test_active2E[indices_next_active2E]))
            y_train_active2E = np.vstack((y_train_active2E, y_test_active2E[indices_next_active2E]))    
            X_test_active2E = np.delete(X_test_active2E, indices_next_active2E, axis=0)
            y_test_active2E = np.delete(y_test_active2E, indices_next_active2E, axis=0)
            XN_test_active2E = np.delete(XN_test_active2E, indices_next_active2E, axis=0)    
            selected_indices_active2E = indices_test_active2E[indices_next_active2E]        
            # print('active2E')
                  
            indices_train_active2E = np.concatenate((indices_train_active2E, selected_indices_active2E))
            indices_train_active2E_gp = np.concatenate((indices_train_active2E_gp, selected_indices_active2E))    
            indices_test_active2E = np.delete(indices_test_active2E, indices_next_active2E, axis=0)   
            # print(len(selected_indices_active2E))
            # print()        
            log_message(f'active2E shape: {X_train_active2E.shape}, {y_train_active2E.shape}, {X_test_active2E.shape}, {y_test_active2E.shape}, {XN_test_active2E.shape}')


            ###
            ##
            ## active3
            ## train model
            if i==0:
                spec2multi_active3 = copy.deepcopy(spec2multi_init)
                mask_active3_clean = ~np.isin(indices_test_active3, noisy_indices)
                X_test_active3_clean = X_test_active3[mask_active3_clean]
                y_test_active3_clean = y_test_active3[mask_active3_clean]                
                
                errors_test_active3 = compute_mse_s2i(spec2multi_active3, norm_0to1_axis1(X_test_active3_clean), y_test_active3_clean, device, multitask=True)
                log_error_active3.append(np.mean(errors_test_active3))
                
            train_dataset_active3 = ImageDataset(norm_0to1_axis1(X_train_active3), y_train_active3)
            val_dataset_active3 = ImageDataset(norm_0to1_axis1(X_val_active3), y_val_active3)
            ## change
            spec2multi_active3 = train_spec2multi(
                spec2multi_active3, train_dataset_active3, val_dataset_active3, device, num_epochs=num_epochs, lr=lr, batch_size=batch_size, wdecay=wdecay1, wloss=wloss1, beta=beta1,
                r_main=r_main, validation=2
            )
            ## errors
            errors_train_active3 = compute_mse_s2i(spec2multi_active3, norm_0to1_axis1(X_train_active3), y_train_active3, device, multitask=True)    
            errors_val_active3 = compute_mse_s2i(spec2multi_active3, norm_0to1_axis1(X_val_active3), y_val_active3, device, multitask=True)        
        
            mask_active3_clean = ~np.isin(indices_test_active3, noisy_indices)
            X_test_active3_clean = X_test_active3[mask_active3_clean]
            y_test_active3_clean = y_test_active3[mask_active3_clean]                
            errors_test_active3 = compute_mse_s2i(spec2multi_active3, norm_0to1_axis1(X_test_active3_clean), y_test_active3_clean, device, multitask=True)    
            log_error_active3.append(np.mean(errors_test_active3))
            ##
            ## train error model
            spec2multi_active3.eval()
            with torch.no_grad():
                X_train_active3_tensor = torch.tensor(X_train_active3, dtype=torch.float32).unsqueeze(1).to(device)
                X_val_active3_tensor = torch.tensor(X_val_active3, dtype=torch.float32).unsqueeze(1).to(device)        
                X_test_active3_tensor = torch.tensor(XN_test_active3, dtype=torch.float32).unsqueeze(1).to(device)            
                X_train_active3_latent = spec2multi_active3.encoder(X_train_active3_tensor).cpu().numpy()        
                X_val_active3_latent = spec2multi_active3.encoder(X_val_active3_tensor).cpu().numpy()
                X_test_active3_latent = spec2multi_active3.encoder(X_test_active3_tensor).cpu().numpy()
                ##
                # Normalize using training set stats
                min_vals = X_train_active3_latent.min(axis=0, keepdims=True)
                max_vals = X_train_active3_latent.max(axis=0, keepdims=True)
                eps = 1e-8
                range_vals = max_vals - min_vals + eps
                X_train_active3_latent = (X_train_active3_latent - min_vals) / range_vals
                X_val_active3_latent   = (X_val_active3_latent - min_vals) / range_vals
                X_test_active3_latent  = (X_test_active3_latent - min_vals) / range_vals            
                ##
            train_error_dataset_active3 = ErrorDataset(X_val_active3_latent, norm_0to1(errors_val_active3))
            val_error_dataset_active3 = ErrorDataset(X_train_active3_latent, norm_0to1(errors_train_active3))        
            if i==0:
                error_model_active3 = copy.deepcopy(error_model_init) ## ErrorModel(latent_dim).to(device)
            ## change
            error_model_active3 = train_error_model(error_model_active3, train_error_dataset_active3, val_error_dataset_active3, device, num_epochs=10*num_epochs, lr=0.5*lr, batch_size=batch_size, 
                                                   wdecay=wdecay2, wloss=wloss2, beta=beta2)
            ## learning/update 
            ############################
            if alpha == 0 and beta == 0 and gamma == 0:
                indices_next_active3 = rng.choice(len(XN_test_active3), num_sample_exp, replace=False)
            else:
                predicted_errors_active3 = norm_0to1(predict_error_model(error_model_active3, X_test_active3_latent, device)) ## for alpha
                distances_active3 = norm_0to1(compute_distances(X_test_active3_latent, X_train_active3_latent)) ## for beta
                sims_active3 = norm_0to1(compute_sims(X_test_active3_latent)) ## for gamma
                acquisition_values_active3 = acquisition_function(predicted_errors_active3, distances_active3, sims_active3, alpha=alpha, beta=beta, gamma=gamma)
                indices_next_active3 = np.argsort(acquisition_values_active3)[-num_sample_exp:]
            ############################    
            X_train_active3 = np.vstack((X_train_active3, XN_test_active3[indices_next_active3]))
            y_train_active3 = np.vstack((y_train_active3, y_test_active3[indices_next_active3]))
            X_test_active3 = np.delete(X_test_active3, indices_next_active3, axis=0)
            y_test_active3 = np.delete(y_test_active3, indices_next_active3, axis=0)
            XN_test_active3 = np.delete(XN_test_active3, indices_next_active3, axis=0)    
            selected_indices_active3 = indices_test_active3[indices_next_active3]        
            # print('active3')
            # print(len(np.intersect1d(noisy_indices, indices_train_active3)), len(np.intersect1d(noisy_indices, selected_indices_active3)), len(np.intersect1d(noisy_indices, indices_test_active3)))                
            indices_train_active3 = np.concatenate((indices_train_active3, selected_indices_active3))
            indices_test_active3 = np.delete(indices_test_active3, indices_next_active3, axis=0)    
            # print(len(np.intersect1d(noisy_indices, indices_train_active3)), len(np.intersect1d(noisy_indices, selected_indices_active3)), len(np.intersect1d(noisy_indices, indices_test_active3)))
            # print(len(selected_indices_active3))
            # print()    
            log_message(f'active3 shape: {X_train_active3.shape}, {y_train_active3.shape}, {X_test_active3.shape}, {y_test_active3.shape}, {XN_test_active3.shape}')
            ###
            
            ##################################
            ## plot: result (test error on clean y)
            ## log
            print([f"{e:.6f}" for e in log_error_active])
            print([f"{e:.6f}" for e in log_error_random])
            print([f"{e:.6f}" for e in log_error_active2A])
            print([f"{e:.6f}" for e in log_error_active2B])
            print([f"{e:.6f}" for e in log_error_active2C])
            print([f"{e:.6f}" for e in log_error_active2D])
            print([f"{e:.6f}" for e in log_error_active2E])

            print([f"{e:.6f}" for e in log_error_active3])
            ##
            d_result[trial_no] = [log_error_active, log_error_random, log_error_active2A, log_error_active2B, log_error_active2C, log_error_active2D, log_error_active2E, log_error_active3]
            d_ratios[trial_no] = [ratios_active, ratios_random, ratios_active2A, ratios_active2B, ratios_active2C, ratios_active2D, ratios_active2E, ratios_active3]            
            ##            
            ##
            # Plot
            plt.figure(figsize=(5, 3))
            plt.plot(log_error_active, label='Active', marker='o', markersize=4, linewidth=1.5, color='blue')
            plt.plot(log_error_random, label='Random', marker='x', markersize=4, linewidth=1.5, color='orange')
            # plt.plot(log_error_active2, label='ActiveQC', marker='+', markersize=4, linewidth=1.5, color='green')
            plt.plot(log_error_active2A, label=f'ActiveQC τ={r2_thresholdA}', marker='+', markersize=4, linewidth=1.5, color='#2d8a2d')
            plt.plot(log_error_active2B, label=f'ActiveQC τ={r2_thresholdB}', marker='+', markersize=4, linewidth=1.5, color='#4CAF50')
            plt.plot(log_error_active2C, label=f'ActiveQC τ={r2_thresholdC}', marker='+', markersize=4, linewidth=1.5, color='#81C784')
            plt.plot(log_error_active2D, label=f'ActiveQC τ={r2_thresholdD}', marker='+', markersize=4, linewidth=1.5, color='#A5D6A7')
            plt.plot(log_error_active2E, label=f'ActiveQC τ={r2_thresholdE}', marker='+', markersize=4, linewidth=1.5, color='#C8E6C9')
            plt.plot(log_error_active3, label='ActiveMT', marker='*', markersize=4, linewidth=1.5, color='purple')    
            plt.xlabel("Exploration Step")
            plt.ylabel("Test Error")
            plt.title(f"")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()    
            plot_path = f"outputs/plots/{exp_no}_result_{i}.png"
            plt.savefig(plot_path, dpi=300)
            plt.show()
            #####################################################################################        
            ## plot: predictions
            ##
            num_samples = 6
            mask_clean = ~np.isin(indices_test, noisy_indices)
            clean_indices_test = np.array(indices_test)[mask_clean]
            chosen_global = random.sample(list(clean_indices_test), num_samples)
            chosen_local = [np.where(indices_test == idx)[0][0] for idx in chosen_global]
            indices = chosen_local
            ##    
            spec2im_active.eval()
            spec2im_random.eval()
            spec2im_active2A.eval()
            spec2im_active2B.eval()
            spec2im_active2C.eval()
            spec2im_active2D.eval()
            spec2im_active2E.eval()

            spec2multi_active3.eval()
            X_test_tensor = torch.tensor(norm_0to1_axis1(X_test[indices]), dtype=torch.float32).to(device)  # [N, D]
            y_test_true = y_test[indices]  # [N, H, W]
            with torch.no_grad():
                pred_active = spec2im_active(X_test_tensor).cpu().numpy()
                pred_random = spec2im_random(X_test_tensor).cpu().numpy()
                pred_active2A = spec2im_active2A(X_test_tensor).cpu().numpy()
                pred_active2B = spec2im_active2B(X_test_tensor).cpu().numpy()
                pred_active2C = spec2im_active2C(X_test_tensor).cpu().numpy()
                pred_active2D = spec2im_active2D(X_test_tensor).cpu().numpy()
                pred_active2E = spec2im_active2E(X_test_tensor).cpu().numpy()                
                pred_active3, _ = spec2multi_active3(X_test_tensor)
                pred_active3 = pred_active3.cpu().numpy()
            # titles = ['GT', 'Active', 'Random', 'ActiveQC', 'ActiveMT']
            titles = [
                'GT', 'Active', 'Random', 
                f'ActiveQC τ={r2_thresholdA}', f'ActiveQC τ={r2_thresholdB}', f'ActiveQC τ={r2_thresholdC}', 
                f'ActiveQC τ={r2_thresholdD}', f'ActiveQC τ={r2_thresholdE}', 
                'ActiveMT'
            ]            
            all_preds = [y_test_true, pred_active, pred_random, pred_active2A, pred_active2B, pred_active2C, pred_active2D, pred_active2E, pred_active3]
            fig, axs = plt.subplots(num_samples, len(titles), figsize=(15, 18))
            axs = axs if num_samples > 1 else [axs]  # handle single sample case
            for row in range(num_samples):
                for col in range(len(titles)):
                    axs[row, col].imshow(all_preds[col][row], cmap='viridis', origin='lower')
                    axs[row, col].axis('off')
                    if row == 0:
                        axs[row, col].set_title(titles[col], fontsize=10)
            plt.tight_layout()
            plot_path = f"outputs/plots/{exp_no}_result_predictions_{i}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            #####################################################################################            
            ########################################################################################################################################
            ## Quality Control cleanUp A
            ########################################################################################################################################
            ##
            ## we could have done this cleanUp part before using active model, 
            ## but just having this cleanUp here shows how quality control can impact 
            ## even almost the similar model used by (active, random, active2A, active3), 
            ## so keep it here to make sure the difference is due to quality control not the model since the model has initially similar error across active/random/active2A/active3
            ##
            if i==0:
                ## val cleanup
                # Step 1: Predict R2 for all active2A validation points
                coords_val_active2A = np.array([coordinates[idx] for idx in indices_val_active2A])
                coords_val_xy = np.array([[c[1], c[0]] for c in coords_val_active2A])  # (x, y)
                r2_pred_val, _ = gpA.predict(coords_val_xy, return_std=True)
                r2_pred_val = norm_0to1(r2_pred_val)    
                # Step 2: Filter by R2 threshold
                keep_mask = r2_pred_val >= r2_thresholdA
                drop_mask = ~keep_mask  # the ones you’re removing
                
                # X_val_active2A = X_val_active2A[keep_mask]
                # y_val_active2A = y_val_active2A[keep_mask]
                # indices_val_active2A = indices_val_active2A[keep_mask]        
        
                # Step 3: Keep only good validation samples
                X_val_keep = X_val_active2A[keep_mask]
                y_val_keep = y_val_active2A[keep_mask]
                indices_val_keep = indices_val_active2A[keep_mask]
            
                # Step 4: Move filtered-out samples to test set
                X_val_drop = X_val_active2A[drop_mask]
                y_val_drop = y_val_active2A[drop_mask]
                indices_val_drop = indices_val_active2A[drop_mask]
            
                # Step 5: Merge with existing test set
                X_test_active2A = np.concatenate([X_test_active2A, X_val_drop], axis=0)
                y_test_active2A = np.concatenate([y_test_active2A, y_val_drop], axis=0)
                XN_test_active2A = np.concatenate([XN_test_active2A, X_val_drop], axis=0)        
                indices_test_active2A = np.concatenate([indices_test_active2A, indices_val_drop], axis=0)
            
                # Step 6: Update validation set with only the kept data
                X_val_active2A = X_val_keep
                y_val_active2A = y_val_keep
                indices_val_active2A = indices_val_keep
        
            ## train cleanup
            # Step 1: Predict R2 for all active2A training points
            coords_train_active2A = np.array([coordinates[idx] for idx in indices_train_active2A])
            coords_train_xy = np.array([[c[1], c[0]] for c in coords_train_active2A])  # (x, y)
            r2_pred_train, _ = gpA.predict(coords_train_xy, return_std=True)
            r2_pred_train = norm_0to1(r2_pred_train)        
            # Step 2: Filter by R2 threshold
            keep_mask = r2_pred_train >= r2_thresholdA
            drop_mask = ~keep_mask  # samples below threshold
            
            # X_train_active2A = X_train_active2A[keep_mask]
            # y_train_active2A = y_train_active2A[keep_mask]
            # indices_train_active2A = indices_train_active2A[keep_mask] 
        
            # Step 3: Keep only good training samples
            X_train_keep = X_train_active2A[keep_mask]
            y_train_keep = y_train_active2A[keep_mask]
            indices_train_keep = indices_train_active2A[keep_mask]
            
            # Step 4: Move dropped ones to test set
            X_train_drop = X_train_active2A[drop_mask]
            y_train_drop = y_train_active2A[drop_mask]
            indices_train_drop = indices_train_active2A[drop_mask]
            
            # Step 5: Merge with existing test set
            X_test_active2A = np.concatenate([X_test_active2A, X_train_drop], axis=0)
            y_test_active2A = np.concatenate([y_test_active2A, y_train_drop], axis=0)
            XN_test_active2A = np.concatenate([XN_test_active2A, X_train_drop], axis=0)    
            indices_test_active2A = np.concatenate([indices_test_active2A, indices_train_drop], axis=0)
            
            # Step 6: Update training set with kept data
            X_train_active2A = X_train_keep
            y_train_active2A = y_train_keep
            indices_train_active2A = indices_train_keep    
        

            ########################################################################################################################################
            ## Quality Control cleanUp B
            ########################################################################################################################################
            ##
            ## we could have done this cleanUp part before using active model, 
            ## but just having this cleanUp here shows how quality control can impact 
            ## even almost the similar model used by (active, random, active2B, active3), 
            ## so keep it here to make sure the difference is due to quality control not the model since the model has initially similar error across active/random/active2B/active3
            ##
            if i==0:
                ## val cleanup
                # Step 1: Predict R2 for all active2B validation points
                coords_val_active2B = np.array([coordinates[idx] for idx in indices_val_active2B])
                coords_val_xy = np.array([[c[1], c[0]] for c in coords_val_active2B])  # (x, y)
                r2_pred_val, _ = gpB.predict(coords_val_xy, return_std=True)
                r2_pred_val = norm_0to1(r2_pred_val)    
                # Step 2: Filter by R2 threshold
                keep_mask = r2_pred_val >= r2_thresholdB
                drop_mask = ~keep_mask  # the ones you’re removing
                
                # X_val_active2B = X_val_active2B[keep_mask]
                # y_val_active2B = y_val_active2B[keep_mask]
                # indices_val_active2B = indices_val_active2B[keep_mask]        
        
                # Step 3: Keep only good validation samples
                X_val_keep = X_val_active2B[keep_mask]
                y_val_keep = y_val_active2B[keep_mask]
                indices_val_keep = indices_val_active2B[keep_mask]
            
                # Step 4: Move filtered-out samples to test set
                X_val_drop = X_val_active2B[drop_mask]
                y_val_drop = y_val_active2B[drop_mask]
                indices_val_drop = indices_val_active2B[drop_mask]
            
                # Step 5: Merge with existing test set
                X_test_active2B = np.concatenate([X_test_active2B, X_val_drop], axis=0)
                y_test_active2B = np.concatenate([y_test_active2B, y_val_drop], axis=0)
                XN_test_active2B = np.concatenate([XN_test_active2B, X_val_drop], axis=0)        
                indices_test_active2B = np.concatenate([indices_test_active2B, indices_val_drop], axis=0)
            
                # Step 6: Update validation set with only the kept data
                X_val_active2B = X_val_keep
                y_val_active2B = y_val_keep
                indices_val_active2B = indices_val_keep
        
            ## train cleanup
            # Step 1: Predict R2 for all active2B training points
            coords_train_active2B = np.array([coordinates[idx] for idx in indices_train_active2B])
            coords_train_xy = np.array([[c[1], c[0]] for c in coords_train_active2B])  # (x, y)
            r2_pred_train, _ = gpB.predict(coords_train_xy, return_std=True)
            r2_pred_train = norm_0to1(r2_pred_train)        
            # Step 2: Filter by R2 threshold
            keep_mask = r2_pred_train >= r2_thresholdB
            drop_mask = ~keep_mask  # samples below threshold
            
            # X_train_active2B = X_train_active2B[keep_mask]
            # y_train_active2B = y_train_active2B[keep_mask]
            # indices_train_active2B = indices_train_active2B[keep_mask] 
        
            # Step 3: Keep only good training samples
            X_train_keep = X_train_active2B[keep_mask]
            y_train_keep = y_train_active2B[keep_mask]
            indices_train_keep = indices_train_active2B[keep_mask]
            
            # Step 4: Move dropped ones to test set
            X_train_drop = X_train_active2B[drop_mask]
            y_train_drop = y_train_active2B[drop_mask]
            indices_train_drop = indices_train_active2B[drop_mask]
            
            # Step 5: Merge with existing test set
            X_test_active2B = np.concatenate([X_test_active2B, X_train_drop], axis=0)
            y_test_active2B = np.concatenate([y_test_active2B, y_train_drop], axis=0)
            XN_test_active2B = np.concatenate([XN_test_active2B, X_train_drop], axis=0)    
            indices_test_active2B = np.concatenate([indices_test_active2B, indices_train_drop], axis=0)
            
            # Step 6: Update training set with kept data
            X_train_active2B = X_train_keep
            y_train_active2B = y_train_keep
            indices_train_active2B = indices_train_keep    
        
            ########################################################################################################################################
            ## Quality Control cleanUp C
            ########################################################################################################################################
            ##
            ## we could have done this cleanUp part before using active model, 
            ## but just having this cleanUp here shows how quality control can impact 
            ## even almost the similar model used by (active, random, active2C, active3), 
            ## so keep it here to make sure the difference is due to quality control not the model since the model has initially similar error across active/random/active2C/active3
            ##
            if i==0:
                ## val cleanup
                # Step 1: Predict R2 for all active2C validation points
                coords_val_active2C = np.array([coordinates[idx] for idx in indices_val_active2C])
                coords_val_xy = np.array([[c[1], c[0]] for c in coords_val_active2C])  # (x, y)
                r2_pred_val, _ = gpC.predict(coords_val_xy, return_std=True)
                r2_pred_val = norm_0to1(r2_pred_val)    
                # Step 2: Filter by R2 threshold
                keep_mask = r2_pred_val >= r2_thresholdC
                drop_mask = ~keep_mask  # the ones you’re removing
                
                # X_val_active2C = X_val_active2C[keep_mask]
                # y_val_active2C = y_val_active2C[keep_mask]
                # indices_val_active2C = indices_val_active2C[keep_mask]        
        
                # Step 3: Keep only good validation samples
                X_val_keep = X_val_active2C[keep_mask]
                y_val_keep = y_val_active2C[keep_mask]
                indices_val_keep = indices_val_active2C[keep_mask]
            
                # Step 4: Move filtered-out samples to test set
                X_val_drop = X_val_active2C[drop_mask]
                y_val_drop = y_val_active2C[drop_mask]
                indices_val_drop = indices_val_active2C[drop_mask]
            
                # Step 5: Merge with existing test set
                X_test_active2C = np.concatenate([X_test_active2C, X_val_drop], axis=0)
                y_test_active2C = np.concatenate([y_test_active2C, y_val_drop], axis=0)
                XN_test_active2C = np.concatenate([XN_test_active2C, X_val_drop], axis=0)        
                indices_test_active2C = np.concatenate([indices_test_active2C, indices_val_drop], axis=0)
            
                # Step 6: Update validation set with only the kept data
                X_val_active2C = X_val_keep
                y_val_active2C = y_val_keep
                indices_val_active2C = indices_val_keep
        
            ## train cleanup
            # Step 1: Predict R2 for all active2C training points
            coords_train_active2C = np.array([coordinates[idx] for idx in indices_train_active2C])
            coords_train_xy = np.array([[c[1], c[0]] for c in coords_train_active2C])  # (x, y)
            r2_pred_train, _ = gpC.predict(coords_train_xy, return_std=True)
            r2_pred_train = norm_0to1(r2_pred_train)        
            # Step 2: Filter by R2 threshold
            keep_mask = r2_pred_train >= r2_thresholdC
            drop_mask = ~keep_mask  # samples below threshold
            
            # X_train_active2C = X_train_active2C[keep_mask]
            # y_train_active2C = y_train_active2C[keep_mask]
            # indices_train_active2C = indices_train_active2C[keep_mask] 
        
            # Step 3: Keep only good training samples
            X_train_keep = X_train_active2C[keep_mask]
            y_train_keep = y_train_active2C[keep_mask]
            indices_train_keep = indices_train_active2C[keep_mask]
            
            # Step 4: Move dropped ones to test set
            X_train_drop = X_train_active2C[drop_mask]
            y_train_drop = y_train_active2C[drop_mask]
            indices_train_drop = indices_train_active2C[drop_mask]
            
            # Step 5: Merge with existing test set
            X_test_active2C = np.concatenate([X_test_active2C, X_train_drop], axis=0)
            y_test_active2C = np.concatenate([y_test_active2C, y_train_drop], axis=0)
            XN_test_active2C = np.concatenate([XN_test_active2C, X_train_drop], axis=0)    
            indices_test_active2C = np.concatenate([indices_test_active2C, indices_train_drop], axis=0)
            
            # Step 6: Update training set with kept data
            X_train_active2C = X_train_keep
            y_train_active2C = y_train_keep
            indices_train_active2C = indices_train_keep    
        
            ########################################################################################################################################
            ## Quality Control cleanUp D
            ########################################################################################################################################
            ##
            ## we could have done this cleanUp part before using active model, 
            ## but just having this cleanUp here shows how quality control can impact 
            ## even almost the similar model used by (active, random, active2D, active3), 
            ## so keep it here to make sure the difference is due to quality control not the model since the model has initially similar error across active/random/active2D/active3
            ##
            if i==0:
                ## val cleanup
                # Step 1: Predict R2 for all active2D validation points
                coords_val_active2D = np.array([coordinates[idx] for idx in indices_val_active2D])
                coords_val_xy = np.array([[c[1], c[0]] for c in coords_val_active2D])  # (x, y)
                r2_pred_val, _ = gpD.predict(coords_val_xy, return_std=True)
                r2_pred_val = norm_0to1(r2_pred_val)    
                # Step 2: Filter by R2 threshold
                keep_mask = r2_pred_val >= r2_thresholdD
                drop_mask = ~keep_mask  # the ones you’re removing
                
                # X_val_active2D = X_val_active2D[keep_mask]
                # y_val_active2D = y_val_active2D[keep_mask]
                # indices_val_active2D = indices_val_active2D[keep_mask]        
        
                # Step 3: Keep only good validation samples
                X_val_keep = X_val_active2D[keep_mask]
                y_val_keep = y_val_active2D[keep_mask]
                indices_val_keep = indices_val_active2D[keep_mask]
            
                # Step 4: Move filtered-out samples to test set
                X_val_drop = X_val_active2D[drop_mask]
                y_val_drop = y_val_active2D[drop_mask]
                indices_val_drop = indices_val_active2D[drop_mask]
            
                # Step 5: Merge with existing test set
                X_test_active2D = np.concatenate([X_test_active2D, X_val_drop], axis=0)
                y_test_active2D = np.concatenate([y_test_active2D, y_val_drop], axis=0)
                XN_test_active2D = np.concatenate([XN_test_active2D, X_val_drop], axis=0)        
                indices_test_active2D = np.concatenate([indices_test_active2D, indices_val_drop], axis=0)
            
                # Step 6: Update validation set with only the kept data
                X_val_active2D = X_val_keep
                y_val_active2D = y_val_keep
                indices_val_active2D = indices_val_keep
        
            ## train cleanup
            # Step 1: Predict R2 for all active2D training points
            coords_train_active2D = np.array([coordinates[idx] for idx in indices_train_active2D])
            coords_train_xy = np.array([[c[1], c[0]] for c in coords_train_active2D])  # (x, y)
            r2_pred_train, _ = gpD.predict(coords_train_xy, return_std=True)
            r2_pred_train = norm_0to1(r2_pred_train)        
            # Step 2: Filter by R2 threshold
            keep_mask = r2_pred_train >= r2_thresholdD
            drop_mask = ~keep_mask  # samples below threshold
            
            # X_train_active2D = X_train_active2D[keep_mask]
            # y_train_active2D = y_train_active2D[keep_mask]
            # indices_train_active2D = indices_train_active2D[keep_mask] 
        
            # Step 3: Keep only good training samples
            X_train_keep = X_train_active2D[keep_mask]
            y_train_keep = y_train_active2D[keep_mask]
            indices_train_keep = indices_train_active2D[keep_mask]
            
            # Step 4: Move dropped ones to test set
            X_train_drop = X_train_active2D[drop_mask]
            y_train_drop = y_train_active2D[drop_mask]
            indices_train_drop = indices_train_active2D[drop_mask]
            
            # Step 5: Merge with existing test set
            X_test_active2D = np.concatenate([X_test_active2D, X_train_drop], axis=0)
            y_test_active2D = np.concatenate([y_test_active2D, y_train_drop], axis=0)
            XN_test_active2D = np.concatenate([XN_test_active2D, X_train_drop], axis=0)    
            indices_test_active2D = np.concatenate([indices_test_active2D, indices_train_drop], axis=0)
            
            # Step 6: Update training set with kept data
            X_train_active2D = X_train_keep
            y_train_active2D = y_train_keep
            indices_train_active2D = indices_train_keep    
        
            ########################################################################################################################################
            ## Quality Control cleanUp E
            ########################################################################################################################################
            ##
            ## we could have done this cleanUp part before using active model, 
            ## but just having this cleanUp here shows how quality control can impact 
            ## even almost the similar model used by (active, random, active2E, active3), 
            ## so keep it here to make sure the difference is due to quality control not the model since the model has initially similar error across active/random/active2E/active3
            ##
            if i==0:
                ## val cleanup
                # Step 1: Predict R2 for all active2E validation points
                coords_val_active2E = np.array([coordinates[idx] for idx in indices_val_active2E])
                coords_val_xy = np.array([[c[1], c[0]] for c in coords_val_active2E])  # (x, y)
                r2_pred_val, _ = gpE.predict(coords_val_xy, return_std=True)
                r2_pred_val = norm_0to1(r2_pred_val)    
                # Step 2: Filter by R2 threshold
                keep_mask = r2_pred_val >= r2_thresholdE
                drop_mask = ~keep_mask  # the ones you’re removing
                
                # X_val_active2E = X_val_active2E[keep_mask]
                # y_val_active2E = y_val_active2E[keep_mask]
                # indices_val_active2E = indices_val_active2E[keep_mask]        
        
                # Step 3: Keep only good validation samples
                X_val_keep = X_val_active2E[keep_mask]
                y_val_keep = y_val_active2E[keep_mask]
                indices_val_keep = indices_val_active2E[keep_mask]
            
                # Step 4: Move filtered-out samples to test set
                X_val_drop = X_val_active2E[drop_mask]
                y_val_drop = y_val_active2E[drop_mask]
                indices_val_drop = indices_val_active2E[drop_mask]
            
                # Step 5: Merge with existing test set
                X_test_active2E = np.concatenate([X_test_active2E, X_val_drop], axis=0)
                y_test_active2E = np.concatenate([y_test_active2E, y_val_drop], axis=0)
                XN_test_active2E = np.concatenate([XN_test_active2E, X_val_drop], axis=0)        
                indices_test_active2E = np.concatenate([indices_test_active2E, indices_val_drop], axis=0)
            
                # Step 6: Update validation set with only the kept data
                X_val_active2E = X_val_keep
                y_val_active2E = y_val_keep
                indices_val_active2E = indices_val_keep
        
            ## train cleanup
            # Step 1: Predict R2 for all active2E training points
            coords_train_active2E = np.array([coordinates[idx] for idx in indices_train_active2E])
            coords_train_xy = np.array([[c[1], c[0]] for c in coords_train_active2E])  # (x, y)
            r2_pred_train, _ = gpE.predict(coords_train_xy, return_std=True)
            r2_pred_train = norm_0to1(r2_pred_train)        
            # Step 2: Filter by R2 threshold
            keep_mask = r2_pred_train >= r2_thresholdE
            drop_mask = ~keep_mask  # samples below threshold
            
            # X_train_active2E = X_train_active2E[keep_mask]
            # y_train_active2E = y_train_active2E[keep_mask]
            # indices_train_active2E = indices_train_active2E[keep_mask] 
        
            # Step 3: Keep only good training samples
            X_train_keep = X_train_active2E[keep_mask]
            y_train_keep = y_train_active2E[keep_mask]
            indices_train_keep = indices_train_active2E[keep_mask]
            
            # Step 4: Move dropped ones to test set
            X_train_drop = X_train_active2E[drop_mask]
            y_train_drop = y_train_active2E[drop_mask]
            indices_train_drop = indices_train_active2E[drop_mask]
            
            # Step 5: Merge with existing test set
            X_test_active2E = np.concatenate([X_test_active2E, X_train_drop], axis=0)
            y_test_active2E = np.concatenate([y_test_active2E, y_train_drop], axis=0)
            XN_test_active2E = np.concatenate([XN_test_active2E, X_train_drop], axis=0)    
            indices_test_active2E = np.concatenate([indices_test_active2E, indices_train_drop], axis=0)
            
            # Step 6: Update training set with kept data
            X_train_active2E = X_train_keep
            y_train_active2E = y_train_keep
            indices_train_active2E = indices_train_keep    
        


            
            ########################################################################################################################################
            ## Quality Control retrain GP A
            ########################################################################################################################################
            # Step 3: Recalculate R2 scores and retrain GP with filtered data
            r2_scores = []
            for idx in indices_train_active2A_gp:
                coord = coordinates[idx]
                avg_r2 = r2_level[idx]    
                r2_scores.append((coord, avg_r2))
            coords_xy = np.array([[c[1], c[0]] for c, _ in r2_scores])
            r2_vals = np.array([s for _, s in r2_scores])
            # gp.fit(coords_xy, r2_vals)
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            gpA_prev = copy.deepcopy(gpA)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gpA.fit(coords_xy, r2_vals)
                    if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                        print("⚠️ GP fit issued ConvergenceWarning — reverting to previous model.")
                        gpA = copy.deepcopy(gpA_prev)
                    else:
                        print("✅ GP fit succeeded without warnings.")
            except Exception as e:
                print(f"⚠️ GP fit failed with error: {e}")
                gpA = copy.deepcopy(gpA_prev)
                
            ##################################
            ## plot: result (test error on clean y)
            coords_all_xy = np.array([[c[1], c[0]] for c in coordinates])
            r2_pred_all, std_pred_all = gpA.predict(coords_all_xy, return_std=True)    
            r2_pred_all = norm_0to1(r2_pred_all)            
            ##
            r2_pred_all_thresh = np.copy(r2_pred_all)
            r2_pred_all_thresh[r2_pred_all_thresh < r2_thresholdA] = np.min(r2_pred_all)
            r2_pred_all = r2_pred_all_thresh
            ##
            # plt.figure(figsize=(5, 5))
            # plt.imshow(full_image, origin='lower', cmap='viridis')
            # sc = plt.scatter(
            #     coords_all_xy[:, 0], coords_all_xy[:, 1],
            #     c=r2_pred_all, cmap='hot', s=10,
            #     norm=Normalize(vmin=r2_pred_all.min(), vmax=r2_pred_all.max())
            # )
            # plt.colorbar(sc, label="R2-score")
            # plt.title("GP Prediction of R2-score Across All Patches")
            # plt.axis("off")
            # plt.tight_layout()
            # plt.savefig(f"outputs/plots/{exp_no}_noise_map_GP_{i}.png", dpi=300)  # <- Save here
            # plt.show()


            
            ########################################################################################################################################
            ## Quality Control retrain GP B
            ########################################################################################################################################
            # Step 3: Recalculate R2 scores and retrain GP with filtered data
            r2_scores = []
            for idx in indices_train_active2B_gp:
                coord = coordinates[idx]
                avg_r2 = r2_level[idx]    
                r2_scores.append((coord, avg_r2))
            coords_xy = np.array([[c[1], c[0]] for c, _ in r2_scores])
            r2_vals = np.array([s for _, s in r2_scores])
            # gp.fit(coords_xy, r2_vals)
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            gpB_prev = copy.deepcopy(gpB)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gpB.fit(coords_xy, r2_vals)
                    if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                        print("⚠️ GP fit issued ConvergenceWarning — reverting to previous model.")
                        gpB = copy.deepcopy(gpB_prev)
                    else:
                        print("✅ GP fit succeeded without warnings.")
            except Exception as e:
                print(f"⚠️ GP fit failed with error: {e}")
                gpB = copy.deepcopy(gpB_prev)
                
            ##################################
            ## plot: result (test error on clean y)
            coords_all_xy = np.array([[c[1], c[0]] for c in coordinates])
            r2_pred_all, std_pred_all = gpB.predict(coords_all_xy, return_std=True)    
            r2_pred_all = norm_0to1(r2_pred_all)            
            ##
            r2_pred_all_thresh = np.copy(r2_pred_all)
            r2_pred_all_thresh[r2_pred_all_thresh < r2_thresholdB] = np.min(r2_pred_all)
            r2_pred_all = r2_pred_all_thresh
            ##
            # plt.figure(figsize=(5, 5))
            # plt.imshow(full_image, origin='lower', cmap='viridis')
            # sc = plt.scatter(
            #     coords_all_xy[:, 0], coords_all_xy[:, 1],
            #     c=r2_pred_all, cmap='hot', s=10,
            #     norm=Normalize(vmin=r2_pred_all.min(), vmax=r2_pred_all.max())
            # )
            # plt.colorbar(sc, label="R2-score")
            # plt.title("GP Prediction of R2-score Across All Patches")
            # plt.axis("off")
            # plt.tight_layout()
            # plt.savefig(f"outputs/plots/{exp_no}_noise_map_GP_{i}.png", dpi=300)  # <- Save here
            # plt.show()
    
            ########################################################################################################################################
            ## Quality Control retrain GP C
            ########################################################################################################################################
            # Step 3: Recalculate R2 scores and retrain GP with filtered data
            r2_scores = []
            for idx in indices_train_active2C_gp:
                coord = coordinates[idx]
                avg_r2 = r2_level[idx]    
                r2_scores.append((coord, avg_r2))
            coords_xy = np.array([[c[1], c[0]] for c, _ in r2_scores])
            r2_vals = np.array([s for _, s in r2_scores])
            # gp.fit(coords_xy, r2_vals)
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            gpC_prev = copy.deepcopy(gpC)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gpC.fit(coords_xy, r2_vals)
                    if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                        print("⚠️ GP fit issued ConvergenceWarning — reverting to previous model.")
                        gpC = copy.deepcopy(gpC_prev)
                    else:
                        print("✅ GP fit succeeded without warnings.")
            except Exception as e:
                print(f"⚠️ GP fit failed with error: {e}")
                gpC = copy.deepcopy(gpC_prev)
                
            ##################################
            ## plot: result (test error on clean y)
            coords_all_xy = np.array([[c[1], c[0]] for c in coordinates])
            r2_pred_all, std_pred_all = gpC.predict(coords_all_xy, return_std=True)    
            r2_pred_all = norm_0to1(r2_pred_all)            
            ##
            r2_pred_all_thresh = np.copy(r2_pred_all)
            r2_pred_all_thresh[r2_pred_all_thresh < r2_thresholdC] = np.min(r2_pred_all)
            r2_pred_all = r2_pred_all_thresh
            ##
            # plt.figure(figsize=(5, 5))
            # plt.imshow(full_image, origin='lower', cmap='viridis')
            # sc = plt.scatter(
            #     coords_all_xy[:, 0], coords_all_xy[:, 1],
            #     c=r2_pred_all, cmap='hot', s=10,
            #     norm=Normalize(vmin=r2_pred_all.min(), vmax=r2_pred_all.max())
            # )
            # plt.colorbar(sc, label="R2-score")
            # plt.title("GP Prediction of R2-score Across All Patches")
            # plt.axis("off")
            # plt.tight_layout()
            # plt.savefig(f"outputs/plots/{exp_no}_noise_map_GP_{i}.png", dpi=300)  # <- Save here
            # plt.show()

            ########################################################################################################################################
            ## Quality Control retrain GP D
            ########################################################################################################################################
            # Step 3: Recalculate R2 scores and retrain GP with filtered data
            r2_scores = []
            for idx in indices_train_active2D_gp:
                coord = coordinates[idx]
                avg_r2 = r2_level[idx]    
                r2_scores.append((coord, avg_r2))
            coords_xy = np.array([[c[1], c[0]] for c, _ in r2_scores])
            r2_vals = np.array([s for _, s in r2_scores])
            # gp.fit(coords_xy, r2_vals)
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            gpD_prev = copy.deepcopy(gpD)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gpD.fit(coords_xy, r2_vals)
                    if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                        print("⚠️ GP fit issued ConvergenceWarning — reverting to previous model.")
                        gpD = copy.deepcopy(gpD_prev)
                    else:
                        print("✅ GP fit succeeded without warnings.")
            except Exception as e:
                print(f"⚠️ GP fit failed with error: {e}")
                gpD = copy.deepcopy(gpD_prev)
                
            ##################################
            ## plot: result (test error on clean y)
            coords_all_xy = np.array([[c[1], c[0]] for c in coordinates])
            r2_pred_all, std_pred_all = gpD.predict(coords_all_xy, return_std=True)    
            r2_pred_all = norm_0to1(r2_pred_all)            
            ##
            r2_pred_all_thresh = np.copy(r2_pred_all)
            r2_pred_all_thresh[r2_pred_all_thresh < r2_thresholdD] = np.min(r2_pred_all)
            r2_pred_all = r2_pred_all_thresh
            ##
            # plt.figure(figsize=(5, 5))
            # plt.imshow(full_image, origin='lower', cmap='viridis')
            # sc = plt.scatter(
            #     coords_all_xy[:, 0], coords_all_xy[:, 1],
            #     c=r2_pred_all, cmap='hot', s=10,
            #     norm=Normalize(vmin=r2_pred_all.min(), vmax=r2_pred_all.max())
            # )
            # plt.colorbar(sc, label="R2-score")
            # plt.title("GP Prediction of R2-score Across All Patches")
            # plt.axis("off")
            # plt.tight_layout()
            # plt.savefig(f"outputs/plots/{exp_no}_noise_map_GP_{i}.png", dpi=300)  # <- Save here
            # plt.show()

            ########################################################################################################################################
            ## Quality Control retrain GP E
            ########################################################################################################################################
            # Step 3: Recalculate R2 scores and retrain GP with filtered data
            r2_scores = []
            for idx in indices_train_active2E_gp:
                coord = coordinates[idx]
                avg_r2 = r2_level[idx]    
                r2_scores.append((coord, avg_r2))
            coords_xy = np.array([[c[1], c[0]] for c, _ in r2_scores])
            r2_vals = np.array([s for _, s in r2_scores])
            # gp.fit(coords_xy, r2_vals)
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            gpE_prev = copy.deepcopy(gpE)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gpE.fit(coords_xy, r2_vals)
                    if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                        print("⚠️ GP fit issued ConvergenceWarning — reverting to previous model.")
                        gpE = copy.deepcopy(gpE_prev)
                    else:
                        print("✅ GP fit succeeded without warnings.")
            except Exception as e:
                print(f"⚠️ GP fit failed with error: {e}")
                gpE = copy.deepcopy(gpE_prev)
                
            ##################################
            ## plot: result (test error on clean y)
            coords_all_xy = np.array([[c[1], c[0]] for c in coordinates])
            r2_pred_all, std_pred_all = gpE.predict(coords_all_xy, return_std=True)    
            r2_pred_all = norm_0to1(r2_pred_all)            
            ##
            r2_pred_all_thresh = np.copy(r2_pred_all)
            r2_pred_all_thresh[r2_pred_all_thresh < r2_thresholdE] = np.min(r2_pred_all)
            r2_pred_all = r2_pred_all_thresh
            ##
            # plt.figure(figsize=(5, 5))
            # plt.imshow(full_image, origin='lower', cmap='viridis')
            # sc = plt.scatter(
            #     coords_all_xy[:, 0], coords_all_xy[:, 1],
            #     c=r2_pred_all, cmap='hot', s=10,
            #     norm=Normalize(vmin=r2_pred_all.min(), vmax=r2_pred_all.max())
            # )
            # plt.colorbar(sc, label="R2-score")
            # plt.title("GP Prediction of R2-score Across All Patches")
            # plt.axis("off")
            # plt.tight_layout()
            # plt.savefig(f"outputs/plots/{exp_no}_noise_map_GP_{i}.png", dpi=300)  # <- Save here
            # plt.show()


        print()

        with open(f"outputs/{exp_no}_d_result.pkl", "wb") as f:
            pickle.dump(d_result, f)

        with open(f"outputs/{exp_no}_d_ratios.pkl", "wb") as f:
            pickle.dump(d_ratios, f)
            
    
    # with open(f"outputs/{exp_no}_d_result.pkl", "rb") as f:
    #     d_result_loaded = pickle.load(f)

    # with open(f"outputs/{exp_no}_d_ratios.pkl", "rb") as f:
    #     d_ratios_loaded = pickle.load(f)    