########################################################################################
## inspired and updated from: https://github.com/ziatdinovmax/im2spec/blob/main/im2spec/models.py
########################################################################################

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class dilated_block(nn.Module):
    """
    Creates a "pyramid" with dilated convolutional
    layers (aka atrous convolutions)
    """
    def __init__(self, ndim: int, input_channels: int, output_channels: int,
                 dilation_values: List, padding_values: List,
                 kernel_size: int = 3, stride: int = 1, lrelu_a: float = 0.01,
                 use_batchnorm: bool = False, dropout_: float = 0) -> None:
        """
        Initializes module parameters
        """
        super(dilated_block, self).__init__()
        conv_ = nn.Conv1d if ndim < 2 else nn.Conv2d
        atrous_module = []
        for idx, (dil, pad) in enumerate(zip(dilation_values, padding_values)):
            input_channels = output_channels if idx > 0 else input_channels
            atrous_module.append(conv_(input_channels,
                                       output_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=pad,
                                       dilation=dil,
                                       bias=True))
            if dropout_ > 0:
                atrous_module.append(nn.Dropout(dropout_))
            atrous_module.append(nn.LeakyReLU(negative_slope=lrelu_a))
            if use_batchnorm:
                if ndim < 2:
                    atrous_module.append(nn.BatchNorm1d(output_channels))
                else:
                    atrous_module.append(nn.BatchNorm2d(output_channels))
        self.atrous_module = nn.Sequential(*atrous_module)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward path
        """
        atrous_layers = []
        for conv_layer in self.atrous_module:
            x = conv_layer(x)
            atrous_layers.append(x.unsqueeze(-1))
        return torch.sum(torch.cat(atrous_layers, dim=-1), dim=-1)
            

class conv_block(nn.Module):
    """
    Creates block of layers each consisting of convolution operation,
    leaky relu and (optionally) dropout and batch normalization
    """
    def __init__(self, ndim: int, nb_layers: int,
                 input_channels: int, output_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 use_batchnorm: bool = False, lrelu_a: float = 0.01,
                 dropout_: float = 0) -> None:
        """
        Initializes module parameters
        """
        super(conv_block, self).__init__()

        conv_ = nn.Conv1d if ndim < 2 else nn.Conv2d
        block = []
        for idx in range(nb_layers):
            input_channels = output_channels if idx > 0 else input_channels
            block.append(conv_(input_channels,
                               output_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding))
            if dropout_ > 0:
                block.append(nn.Dropout(dropout_))
            block.append(nn.LeakyReLU(negative_slope=lrelu_a))
            if use_batchnorm:
                if ndim < 2:
                    block.append(nn.BatchNorm1d(output_channels))
                else:
                    block.append(nn.BatchNorm2d(output_channels))
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward path
        """
        output = self.block(x)
        return output


class im2spec(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size: Tuple[int, int],
                 target_size: int,
                 latent_dim: int = 10,
                 nb_filters_enc: int = 64,
                 nb_filters_dec: int = 64,
                 dropout: float = 0.0) -> None:
        super(im2spec, self).__init__()
        self.n, self.m = feature_size
        self.ts = target_size
        self.e_filt = nb_filters_enc
        self.d_filt = nb_filters_dec
        # Encoder params
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.enc_fc = nn.Linear(self.e_filt * self.n * self.m, latent_dim)
        # Decoder params
        self.dec_fc = nn.Linear(latent_dim, self.d_filt*self.ts)
        self.dec_atrous = dilated_block(
            ndim=1, input_channels=self.d_filt, output_channels=self.d_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_conv = conv_block(
            ndim=1, nb_layers=1,
            input_channels=self.d_filt, output_channels=1,
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_out = nn.Conv1d(1, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.final_act = nn.Sigmoid()  # Ensures output stays within [0, 1]

    def encoder(self, features: torch.Tensor) -> torch.Tensor:
        x = self.enc_conv(features)
        x = x.reshape(-1, self.e_filt * self.m * self.n)
        x = self.dropout(self.enc_fc(x))  # Apply dropout here
        return x        

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.dec_fc(encoded))  # Apply dropout here
        x = x.reshape(-1, self.d_filt, self.ts)
        x = self.dec_atrous(x)
        x = self.dropout(self.dec_conv(x))  # Apply dropout before final conv
        return self.final_act(self.dec_out(x)) ## self.dec_out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        x = x.unsqueeze(1)        
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""
        with torch.no_grad():  # Disable gradient calculation for inference
            return self.forward(x)
            

class spec2im(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size: int,
                 target_size: Tuple[int, int],
                 latent_dim: int = 10,
                 nb_filters_enc: int = 64,
                 nb_filters_dec: int = 64,
                 dropout: float = 0.0) -> None:
        super(spec2im, self).__init__()
        self.n, self.m = target_size
        self.fs = feature_size
        self.e_filt = nb_filters_enc
        self.d_filt = nb_filters_dec
        # Encoder params
        self.enc_conv = conv_block(
            ndim=1, nb_layers=4,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.enc_fc = nn.Linear(self.e_filt * self.fs, latent_dim)
        # Decoder params
        self.dec_fc = nn.Linear(latent_dim, self.d_filt * (self.n // 4) * (self.m // 4))
        self.dec_conv_1 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=self.d_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_conv_2 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=self.d_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_atrous = dilated_block(
            ndim=2, input_channels=self.d_filt, output_channels=self.d_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_conv_3 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=1,
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_out = nn.Conv2d(1, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.final_act = nn.Sigmoid()  # Ensures output stays within [0, 1]

    def encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the imput signal into a latent vector
        """
        x = self.enc_conv(features)
        x = x.reshape(-1, self.e_filt * self.fs)
        x = self.dropout(self.enc_fc(x))
        return x
    
    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 2D image from the embedded features
        """
        x = self.dropout(self.dec_fc(encoded))
        x = x.reshape(-1, self.d_filt, self.n//4, self.m//4)
        x = self.dropout(self.dec_conv_1(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.dec_conv_2(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.dec_atrous(x)
        x = self.dropout(self.dec_conv_3(x))
        return self.final_act(self.dec_out(x))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        x = x.unsqueeze(1)                
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""
        with torch.no_grad():  # Disable gradient calculation for inference
            return self.forward(x)


class im2multi(nn.Module):
    """
    Multitask model: Shared encoder with two decoders —
    one for spectra prediction and one for image reconstruction.
    """
    def __init__(self,
                 feature_size: Tuple[int, int],
                 target_size: int,
                 latent_dim: int = 10,
                 nb_filters_enc: int = 64,
                 nb_filters_dec: int = 64,
                 dropout: float = 0.0) -> None:
        super(im2multi, self).__init__()
        self.n, self.m = feature_size
        self.ts = target_size
        self.latent_dim = latent_dim
        self.e_filt = nb_filters_enc
        self.d_filt = nb_filters_dec

        # Shared Encoder
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.enc_fc = nn.Linear(self.e_filt * self.n * self.m, latent_dim)

        # Spectra Decoder (1D)
        self.spec_fc = nn.Linear(latent_dim, self.d_filt * self.ts)
        self.spec_atrous = dilated_block(
            ndim=1, input_channels=self.d_filt, output_channels=self.d_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, use_batchnorm=True)
        self.spec_conv = conv_block(
            ndim=1, nb_layers=1,
            input_channels=self.d_filt, output_channels=1,
            lrelu_a=0.1, use_batchnorm=True)
        self.spec_out = nn.Conv1d(1, 1, 1)

        # Image Decoder (2D)
        self.img_fc = nn.Linear(latent_dim, self.d_filt * self.n * self.m)
        self.img_conv_1 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=self.d_filt,
            padding=1,  # Added
            lrelu_a=0.1, use_batchnorm=True)
        self.img_conv_2 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=self.d_filt,
            padding=1,  # Added
            lrelu_a=0.1, use_batchnorm=True)
        self.img_atrous = dilated_block(
            ndim=2, input_channels=self.d_filt, output_channels=self.d_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, use_batchnorm=True)
        self.img_conv_3 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=1,
            padding=1,  # Added
            lrelu_a=0.1, use_batchnorm=True)
        self.img_out = nn.Conv2d(1, 1, 1)

        self.dropout = nn.Dropout(dropout)
        self.final_act = nn.Sigmoid()  # Ensures output stays within [0, 1]
        
    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_conv(x)
        x = x.reshape(-1, self.e_filt * self.n * self.m)
        return self.dropout(self.enc_fc(x))

    def decode_spectra(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.spec_fc(z))
        x = x.reshape(-1, self.d_filt, self.ts)
        x = self.spec_atrous(x)
        x = self.dropout(self.spec_conv(x))
        return self.final_act(self.spec_out(x))

    def decode_image(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.img_fc(z)) ## x = self.img_fc(z)
        x = x.reshape(-1, self.d_filt, self.n, self.m)
        x = self.dropout(self.img_conv_1(x)) ## x = self.img_conv_1(x)
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.img_conv_2(x)
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.img_atrous(x)
        x = self.dropout(self.img_conv_3(x)) ## x = self.img_conv_3(x)
        return self.final_act(self.img_out(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # [B, 1, H, W]
        z = self.encoder(x)
        spectra = self.decode_spectra(z).squeeze(1) # [B, 121]
        recon_img = self.decode_image(z).squeeze(1) # [B, H, W]
        return spectra, recon_img

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict spectra and reconstructed image from input patch"""
        self.eval()  # Set model to eval mode (optional but recommended)
        with torch.no_grad():
            return self.forward(x)


class spec2multi(nn.Module):
    """
    Multitask model: Shared encoder with two decoders —
    one for image reconstruction and one for spectra reconstruction.
    """
    def __init__(self,
                 feature_size: int,              # input spectra size
                 target_size: Tuple[int, int],   # image patch size
                 latent_dim: int = 10,
                 nb_filters_enc: int = 64,
                 nb_filters_dec: int = 64,
                 dropout: float = 0.0) -> None:
        super(spec2multi, self).__init__()
        self.fs = feature_size
        self.n, self.m = target_size
        self.latent_dim = latent_dim
        self.e_filt = nb_filters_enc
        self.d_filt = nb_filters_dec

        # Shared Encoder (1D conv on spectra)
        self.enc_conv = conv_block(
            ndim=1, nb_layers=4,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.enc_fc = nn.Linear(self.e_filt * self.fs, latent_dim)

        # Image Decoder (2D)
        self.img_fc = nn.Linear(latent_dim, self.d_filt * (self.n // 4) * (self.m // 4))
        self.img_conv_1 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=self.d_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.img_conv_2 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=self.d_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.img_atrous = dilated_block(
            ndim=2, input_channels=self.d_filt, output_channels=self.d_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, use_batchnorm=True)
        self.img_conv_3 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=1,
            lrelu_a=0.1, use_batchnorm=True)
        self.img_out = nn.Conv2d(1, 1, 1)

        # Spectra Decoder (1D reconstruction)
        self.spec_fc = nn.Linear(latent_dim, self.d_filt * self.fs)
        self.spec_atrous = dilated_block(
            ndim=1, input_channels=self.d_filt, output_channels=self.d_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, use_batchnorm=True)
        self.spec_conv = conv_block(
            ndim=1, nb_layers=1,
            input_channels=self.d_filt, output_channels=1,
            lrelu_a=0.1, use_batchnorm=True)
        self.spec_out = nn.Conv1d(1, 1, 1)

        self.dropout = nn.Dropout(dropout)
        self.final_act = nn.Sigmoid()

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_conv(x)
        x = x.reshape(-1, self.e_filt * self.fs)
        x = self.dropout(self.enc_fc(x))
        return x

    def decode_image(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.img_fc(z))
        x = x.reshape(-1, self.d_filt, self.n // 4, self.m // 4)
        x = self.dropout(self.img_conv_1(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.img_conv_2(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.img_atrous(x)
        x = self.dropout(self.img_conv_3(x))
        return self.final_act(self.img_out(x))
    
    def decode_spectra(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.spec_fc(z))
        x = x.reshape(-1, self.d_filt, self.fs)
        x = self.spec_atrous(x)
        x = self.dropout(self.spec_conv(x))
        return self.final_act(self.spec_out(x))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # [B, 1, fs]
        z = self.encoder(x)
        recon_img = self.decode_image(z).squeeze(1)   # [B, n, m]
        recon_spec = self.decode_spectra(z).squeeze(1)  # [B, fs]
        return recon_img, recon_spec

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict image and spectra from input spectra"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class ErrorModel(nn.Module):
    def __init__(self, latent_dim, conv1_channels, conv2_channels, output_dim=1):
        super(ErrorModel, self).__init__()
        self.conv1 = nn.Conv1d(1, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        if output_dim == 1:
            self.fc1 = nn.Linear(conv2_channels * latent_dim, conv2_channels * 2)  # Increased neurons
            self.fc2 = nn.Linear(conv2_channels * 2, conv2_channels)
            self.fc3 = nn.Linear(conv2_channels, conv1_channels)
            self.fc4 = nn.Linear(conv1_channels, output_dim)  
        else:
            self.fc1 = nn.Linear(conv2_channels * latent_dim, conv2_channels * 2 * 2)  # Increased neurons
            self.fc2 = nn.Linear(conv2_channels * 2 * 2, conv2_channels * 2)
            self.fc3 = nn.Linear(conv2_channels * 2, conv1_channels * 2)
            self.fc4 = nn.Linear(conv1_channels * 2, output_dim)  
        self.relu = nn.ReLU()
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim for Conv1D
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.final_act(self.fc4(x))  
