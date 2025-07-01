# Advanced Audio Deepfake/SingFake Generator using Multiple Variational Autoencoder (VAE) Variants
![Copy of APSIPA (8)](https://github.com/user-attachments/assets/d30043c0-af5b-4def-98b5-7304efaca6dd)

This project implements an end-to-end pipeline for generating deepfake audio using a Variational Autoencoder (VAE) and its enhanced variants (`bsrbfkan-gelu-gated-vae`, `bsrbfkan-vae`, `chebyshevkanlinear-gelu-gated-vae`, `chebyshevkanlinear-vae`, `gelu-gated-vae`, `vae-simple`, `wavkan-gelu-gated-vae`, `wavkan-vae`), trained on Mel spectrograms. The system processes audio data, trains models, generates deepfake variations, and evaluates results with visualizations and metrics. Caching mechanisms enhance performance by reducing redundant computations.

## Features

- **Audio Preprocessing**: Converts audio to Mel spectrograms with caching to avoid reprocessing.
- **VAE Models**: Supports a standard VAE and enhanced variants with specialized layers.
- **Deepfake Generation**: Produces high-quality deepfakes with controlled noise in the latent space.
- **Evaluation Metrics**: Computes MSE, Cosine Similarity, and Pearson Correlation.
- **Visualization**: Displays Mel spectrograms and comparison plots.
- **Audio Output**: Saves audio as WAV files.
- **Performance Optimization**: Includes caching for preprocessing and model training.


## Installation

### Prerequisites
- Python 3.8+
- PyTorch (`torch`, `torchvision`, `torchaudio`)
- Additional libraries: `librosa`, `soundfile`, `matplotlib`, `numpy`, `scikit-learn`, `tqdm`, `Pillow`

### Install Dependencies
Run the following command to install required libraries:
```bash
pip install torch torchvision torchaudio librosa soundfile matplotlib numpy scikit-learn tqdm Pillow
```

### CUDA Configuration
The code automatically detects and uses CUDA if available. Ensure CUDA is installed for GPU acceleration. The environment variable `PYTORCH_CUDA_ALLOC_CONF` is set to `expandable_segments:True` to optimize memory allocation.

## Usage

1. **Prepare Data**:
   - Place Mel spectrogram images (PNG, JPG, JPEG) in a directory (e.g., `/kaggle/input/vae128-ctrsdd/spec_128`).
   - Ensure spectrograms are grayscale and compatible with the target shape (default: 128x128).

2. **Run the Script**:
   - Execute the main script to process data, train the VAE, generate deepfakes, and evaluate results:
     ```bash
     python main.py
     ```
   - The script processes training, validation, and test sets, trains the VAE for up to 100 epochs, and generates deepfake audio for 11 test samples with configurable noise levels.

3. **Output**:
   - **Model Checkpoints**: Saved in the `models/` directory (`best_vae_model.pth`).
   - **Audio Files**: Original and generated WAV files (e.g., `original_reference_improved.wav`, `high_quality_deepfake_X_noise_Y.wav`).
   - **Visualizations**: Plots of Mel spectrograms, training loss, and comparison between original and generated spectrograms.
## All codes are present in IPYND Folder
![bbae818f-51d1-4273-9d8b-f68170f8176f](https://github.com/user-attachments/assets/e152066d-3bf9-4e75-a9b5-23ef94e869b0)
Now below is code snippet with all class
## Code Snippets
## RadialBasisFunction Class
Implements a radial basis function layer for BSRBFKANLinear.
```python
class RadialBasisFunction(nn.Module):
    def __init__(self, low, high, num_centers):
        super().__init__()
        self.register_buffer("centers", torch.linspace(low, high, num_centers))
        self.gamma = 1.0

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.exp(-self.gamma * (x - self.centers) ** 2)
```
## BSRBFKANLinear Class
Combines B-splines and radial basis functions for enhanced linear transformation.
```python
class BSRBFKANLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 3, spline_order: int = 2, base_activation=nn.SiLU, grid_range=[-1.0, 1.0]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()
        self.layernorm = nn.LayerNorm(input_dim)

        self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        self.spline_weight = nn.Parameter(torch.empty(output_dim, input_dim * (grid_size + spline_order)))
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size + spline_order)

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(input_dim, -1).contiguous()
        self.register_buffer("grid", grid)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.input_dim
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]) + ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        base_out = F.linear(self.base_activation(x), self.base_weight)
        bspline_out = self.b_splines(x).view(x.size(0), -1)
        rbf_out = self.rbf(x).view(x.size(0), -1)
        combined_basis = bspline_out + rbf_out
        nonlinear_out = F.linear(combined_basis, self.spline_weight)
        return base_out + nonlinear_out
```
## GeluGatedLayer Class
Implements a GELU-activated gated layer for VAE enhancements.
```python
class GeluGatedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(GeluGatedLayer, self).__init__()
        self.input_linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.activation = nn.GELU()

    def forward(self, src):
        output = self.activation(self.input_linear(src))
        return output
```
## ChebyshevKANLinear Class
Utilizes Chebyshev polynomials for advanced linear transformations with normalization.
```python
class ChebyshevKANLinear(nn.Module):
    def __init__(self, in_features, out_features, degree=5, scale_base=1.0, scale_cheb=1.0, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.cheb_weight = nn.Parameter(torch.Tensor(out_features, in_features, degree + 1))
        self.scale_base = scale_base
        self.scale_cheb = scale_cheb
        self.layernorm = nn.LayerNorm(in_features).to(self.device)
        self.base_activation = nn.PReLU().to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_weight, gain=0.1)
        nn.init.xavier_uniform_(self.cheb_weight, gain=0.1)

    def chebyshev_polynomials(self, x):
        x = x.to(self.device)
        T = [torch.ones_like(x), x]
        for n in range(2, self.degree + 1):
            Tn = 2 * x * T[-1] - T[-2]
            T.append(Tn.clamp(-10, 10))
        return torch.stack(T, dim=-1)

    def forward(self, x):
        x = self.layernorm(x)
        x = x.clamp(-1, 1)
        x = x.to(self.device)
        base_out = F.linear(self.base_activation(x), self.base_weight)
        cheb_terms = self.chebyshev_polynomials(x)
        cheb_proj = torch.einsum('bid,oif->bo', cheb_terms, self.cheb_weight)
        output = self.scale_base * base_out + self.scale_cheb * cheb_proj
        if torch.isnan(output).any():
            print("Warning: NaN detected in ChebyshevKANLinear output")
        return output
```
## WavKANLinear Class
Implements a wavelet-based KAN linear layer with chunked processing.
```python
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='dog', chunk_size=1024):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.chunk_size = chunk_size

        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = nn.PReLU()

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x
        batch_size = x_expanded.size(0)
        wavelet_output = torch.zeros(batch_size, self.out_features, device=x.device)
        for i in range(0, self.in_features, self.chunk_size):
            end_idx = min(i + self.chunk_size, self.in_features)
            x_chunk = x_expanded[:, :, i:end_idx]
            scale_chunk = self.scale[:, i:end_idx].unsqueeze(0)
            x_scaled = x_chunk / (scale_chunk + 1e-8)
            if self.wavelet_type == 'dog':
                x_squared = x_scaled ** 2
                wavelet = -x_scaled * torch.exp(-0.5 * x_squared)
                del x_squared
            else:
                raise ValueError("Only 'dog' wavelet is supported")
            wavelet_weighted = wavelet * self.wavelet_weights[:, i:end_idx].unsqueeze(0)
            wavelet_output += wavelet_weighted.sum(dim=2)
            del x_chunk, scale_chunk, wavelet, wavelet_weighted
            torch.cuda.empty_cache()
        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        base_output = F.linear(self.base_activation(x), self.weight1)
        combined_output = wavelet_output + base_output
        del wavelet_output, base_output
        torch.cuda.empty_cache()
        return combined_output
```

## Architecture Overview

- **Encoder**: Transforms Mel spectrograms into a latent space using convolutional layers, with optional custom layers for enhanced variants. Outputs mean and log-variance for variational sampling.
  
![APSIPA (10) (1)](https://github.com/user-attachments/assets/fe88840f-6d61-46cd-a5b5-7262edc7b92a)


- **Decoder**: Reconstructs spectrograms from the latent space using transposed convolutions, ensuring output matches the input shape.
  
![APSIPA (10)](https://github.com/user-attachments/assets/3928856d-4536-4963-9baf-27db826faa19)


- **Bottleneck**: Compresses data into a latent representation (default 256 dimensions), enabling diverse deepfake generation via noise and KL divergence regularization.

## ImprovedAudioVAE Class (Base VAE)
Defines the base VAE architecture used across variants.
```python
class ImprovedAudioVAE(nn.Module):
    def __init__(self, input_shape, conv_filters=(16, 32, 64, 128), conv_kernels=(3, 3, 3, 3),
                 conv_strides=(1, 2, 2, 2), latent_dim=256, dropout_rate=0.3, custom_layer=None):
        super(ImprovedAudioVAE, self).__init__()
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.num_conv_layers = len(conv_filters)
        self.shape_before_bottleneck = None
        self.custom_layer = custom_layer

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def _build_encoder(self):
        layers = []
        in_channels = 1
        for i, (filters, kernel, stride) in enumerate(zip(self.conv_filters, self.conv_kernels, self.conv_strides)):
            layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel, stride=stride, padding=kernel // 2),
                nn.ReLU(),
                nn.BatchNorm2d(filters),
                nn.Dropout(self.dropout_rate)
            ])
            in_channels = filters

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.input_shape)
            x = dummy_input
            for layer in layers:
                x = layer(x)
            self.shape_before_bottleneck = x.shape[1:]

        flat_dim = np.prod(self.shape_before_bottleneck)
        layers.append(nn.Flatten())
        if self.custom_layer:
            layers.append(self.custom_layer(flat_dim, self.latent_dim))
        else:
            self.mu = nn.Linear(flat_dim, self.latent_dim)
            self.logvar = nn.Linear(flat_dim, self.latent_dim)
        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = []
        num_neurons = np.prod(self.shape_before_bottleneck)
        
        layers.extend([
            nn.Linear(self.latent_dim, num_neurons),
            nn.ReLU(),
            nn.Unflatten(1, self.shape_before_bottleneck)
        ])

        in_channels = self.conv_filters[-1]
        for i in reversed(range(1, self.num_conv_layers)):
            layers.extend([
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.conv_filters[i-1], kernel_size=self.conv_kernels[i], stride=self.conv_strides[i], padding=self.conv_kernels[i] // 2, output_padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(self.conv_filters[i-1]),
                nn.Dropout(self.dropout_rate)
            ])
            in_channels = self.conv_filters[i-1]

        layers.extend([
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=1, kernel_size=self.conv_kernels[0], stride=self.conv_strides[0], padding=self.conv_kernels[0] // 2, output_padding=0),
            nn.Sigmoid()
        ])
        return nn.Sequential(*layers)

    def encode(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        h = self.encoder(x)
        if self.custom_layer:
            mu = h
            logvar = torch.zeros_like(h)
        else:
            mu = self.mu(h)
            logvar = self.logvar(h)
        return mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        recon_x = torch.nn.functional.interpolate(x, size=self.input_shape, mode='bilinear', align_corners=False)
        return recon_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
```

## AudioProcessor Class
Handles Mel spectrogram conversion and visualization.
```python
class AudioProcessor:
    def __init__(self, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def mel_spectrogram_to_audio(self, mel_spec_db):
        mel_spec = librosa.db_to_power(mel_spec_db)
        stft = librosa.feature.inverse.mel_to_stft(mel_spec, sr=self.sr, n_fft=self.n_fft)
        audio = librosa.griffinlim(stft, hop_length=self.hop_length)
        return audio

    def plot_mel_spectrogram(self, mel_spec_db, title="Mel Spectrogram"):
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(mel_spec_db, sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()
```

## AudioDataset Class
PyTorch Dataset for loading Mel spectrograms.
```python
class AudioDataset(Dataset):
    def __init__(self, mel_spectrograms):
        self.mel_spectrograms = mel_spectrograms

    def __len__(self):
        return len(self.mel_spectrograms)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.mel_spectrograms[idx])
```

## AudioDeepfakeGenerator Class
Orchestrates the full workflow for deepfake generation.
```python
class AudioDeepfakeGenerator:
    def __init__(self, height=128, width=128, model_type="vae-simple"):
        self.processor = AudioProcessor(n_mels=height, sr=22050, n_fft=2048, hop_length=512)
        self.vae = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.target_shape = (height, width)
        self.model_type = model_type
        self._init_vae()

    def _init_vae(self):
        input_shape_vae = self.target_shape
        custom_layer = None
        if self.model_type == "bsrbfkan-gelu-gated-vae":
            custom_layer = nn.Sequential(BSRBFKANLinear(np.prod(input_shape_vae), 256), GeluGatedLayer(256, 256))
        elif self.model_type == "bsrbfkan-vae":
            custom_layer = BSRBFKANLinear(np.prod(input_shape_vae), 256)
        elif self.model_type == "chebyshevkanlinear-gelu-gated-vae":
            custom_layer = nn.Sequential(ChebyshevKANLinear(np.prod(input_shape_vae), 256), GeluGatedLayer(256, 256))
        elif self.model_type == "chebyshevkanlinear-vae":
            custom_layer = ChebyshevKANLinear(np.prod(input_shape_vae), 256)
        elif self.model_type == "gelu-gated-vae":
            custom_layer = GeluGatedLayer(np.prod(input_shape_vae), 256)
        elif self.model_type == "wavkan-gelu-gated-vae":
            custom_layer = nn.Sequential(KANLinear(np.prod(input_shape_vae), 256), GeluGatedLayer(256, 256))
        elif self.model_type == "wavkan-vae":
            custom_layer = KANLinear(np.prod(input_shape_vae), 256)
        self.vae = ImprovedAudioVAE(input_shape_vae, custom_layer=custom_layer).to(self.device)

    def preprocess_mel_spectrograms(self, file_data):
        mel_spectrograms = []
        successful_files = 0
        print(f"Processing {len(file_data)} mel spectrogram image files...")

        for file_path, mel_spec in file_data:
            try:
                if mel_spec.shape[0] > self.target_shape[0]:
                    mel_spec = mel_spec[:self.target_shape[0], :]
                elif mel_spec.shape[0] < self.target_shape[0]:
                    pad_height = self.target_shape[0] - mel_spec.shape[0]
                    mel_spec = np.pad(mel_spec, ((0, pad_height), (0, 0)), 'constant', constant_values=mel_spec.min())

                if mel_spec.shape[1] > self.target_shape[1]:
                    mel_spec = mel_spec[:, :self.target_shape[1]]
                elif mel_spec.shape[1] < self.target_shape[1]:
                    pad_width = self.target_shape[1] - mel_spec.shape[1]
                    mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), 'constant', constant_values=mel_spec.min())

                if mel_spec.shape != self.target_shape:
                    print(f"Warning: Shape mismatch for {os.path.basename(file_path)}: got {mel_spec.shape}, expected {self.target_shape}. Skipping.")
                    continue

                mel_min, mel_max = np.percentile(mel_spec, [1, 99])
                if mel_max <= mel_min:
                    print(f"Warning: Invalid mel spectrogram range for {os.path.basename(file_path)}. Skipping.")
                    continue

                mel_norm = np.clip((mel_spec - mel_min) / (mel_max - mel_min + 1e-8), 0, 1)
                mel_spectrograms.append(mel_norm)
                successful_files += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        if len(mel_spectrograms) > 0:
            mel_spectrograms = np.stack(mel_spectrograms)[:, None, :, :]
            print(f"Successfully processed {len(mel_spectrograms)} files")
            print(f"Final data shape: {mel_spectrograms.shape}")
            if mel_spectrograms.shape[2:] != self.target_shape:
                raise ValueError(f"Stacked spectrograms have incorrect shape: {mel_spectrograms.shape[2:]}, expected {self.target_shape}")
            return mel_spectrograms
        else:
            raise ValueError("No mel spectrogram image files could be processed successfully!")

    def train_vae(self, mel_spectrograms, epochs=100, batch_size=32, learning_rate=5e-5, beta=0.02, early_stop_patience=50, accum_steps=32):
        input_shape_vae = mel_spectrograms.shape[2:]
        print(f"VAE input shape (H, W): {input_shape_vae}")

        dataset = AudioDataset(mel_spectrograms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        optimizer = optim.AdamW(self.vae.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = GradScaler()
        best_loss = float('inf')
        patience_counter = 0

        self.vae.train()
        train_losses = []

        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')

            optimizer.zero_grad(set_to_none=True)
            for batch_idx, data in enumerate(progress_bar):
                data = data.to(self.device, non_blocking=True)

                with autocast():
                    recon_batch, mu, logvar = self.vae(data)
                    recon_loss, kl_loss = self.improved_vae_loss(recon_batch, data, mu, logvar, beta=beta)
                    loss = (recon_loss + kl_loss) / accum_steps

                scaler.scale(loss).backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item() * accum_steps
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

                progress_bar.set_postfix({
                    'loss': f'{loss.item() * accum_steps:.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })

                del data, recon_batch, mu, logvar, loss, recon_loss, kl_loss
                gc.collect()
                torch.cuda.empty_cache()

            avg_loss = total_loss / len(dataloader)
            avg_recon_loss = total_recon_loss / len(dataloader)
            avg_kl_loss = total_kl_loss / len(dataloader)

            train_losses.append(avg_loss)
            scheduler.step()

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                os.makedirs("models", exist_ok=True)
                torch.save(self.vae.state_dict(), f'models/best_{self.model_type}_vae.pth')
                print(f"Best {self.model_type} model saved.")
            else:
                patience_counter += 1

            if epoch % 10 == 0 or patience_counter == 0:
                print(f'Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | KL: {avg_kl_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f} | Patience: {patience_counter}/{early_stop_patience}')

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if os.path.exists(f'models/best_{self.model_type}_vae.pth'):
            self.vae.load_state_dict(torch.load(f'models/best_{self.model_type}_vae.pth'))
            print(f"Best {self.model_type} model loaded.")
        else:
            print(f"Warning: Best {self.model_type} model checkpoint not found.")

        print(f"{self.model_type} VAE training completed!")
        self.plot_training_curve(train_losses)

    def improved_vae_loss(self, recon_x, x, mu, logvar, beta):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss, beta * kl_loss

    def plot_training_curve(self, train_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_type} VAE Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_deepfake(self, original_mel_spec, noise_levels=[0.0, 0.1, 0.2]):
        if self.vae is None:
            print(f"Error: {self.model_type} VAE model not trained. Please train the model first.")
            return None

        self.vae.eval()
        deepfake_results = []

        with torch.no_grad():
            if original_mel_spec.shape != self.target_shape:
                print(f"Warning: Input mel shape {original_mel_spec.shape} does not match VAE target shape {self.target_shape}. Resizing.")
                original_mel_spec = np.resize(original_mel_spec, self.target_shape)

            mel_min, mel_max = np.percentile(original_mel_spec, [1, 99])
            if mel_max <= mel_min:
                print("Error: Invalid mel spectrogram range for generation.")
                return original_mel_spec

            mel_norm = np.clip((original_mel_spec - mel_min) / (mel_max - mel_min + 1e-8), 0, 1)
            input_tensor = torch.FloatTensor(mel_norm[None, None, :, :]).to(self.device, non_blocking=True)

            mu, logvar = self.vae.encode(input_tensor)
            for noise_level in noise_levels:
                noise = torch.randn_like(mu, device=self.device) * noise_level
                z_modified = mu + noise
                fake_mel_norm = self.vae.decode(z_modified).cpu().numpy().squeeze()
                fake_mel_norm = np.clip(fake_mel_norm, 0, 1)
                fake_mel = fake_mel_norm * (mel_max - mel_min) + mel_min
                deepfake_results.append((fake_mel, noise_level))

            del input_tensor, mu, logvar, noise, z_modified, fake_mel_norm
            gc.collect()
            torch.cuda.empty_cache()

        return deepfake_results

    def evaluate_model(self, test_spectrograms, beta=0.02):
        if self.vae is None:
            print(f"Error: {self.model_type} VAE model not trained for evaluation.")
            return None

        self.vae.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_samples = 0
        with torch.no_grad():
            dataloader = DataLoader(AudioDataset(test_spectrograms), batch_size=2, shuffle=False, num_workers=0)
            for batch_idx, batch in enumerate(dataloader):
                batch_tensor = batch.to(self.device, non_blocking=True)
                with autocast():
                    recon_batch, mu, logvar = self.vae(batch_tensor)
                    recon_loss, kl_loss = self.improved_vae_loss(recon_batch, batch_tensor, mu, logvar, beta=beta)
                total_loss += (recon_loss + kl_loss).item() * len(batch)
                total_recon_loss += recon_loss.item() * len(batch)
                total_kl_loss += kl_loss.item() * len(batch)
                total_samples += len(batch)
                del batch_tensor, recon_batch, mu, logvar, recon_loss, kl_loss
                gc.collect()
                torch.cuda.empty_cache()

        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_recon_loss = total_recon_loss / total_samples
            avg_kl_loss = total_kl_loss / total_samples
            print(f"Evaluation Loss for {self.model_type}: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f})")
            return avg_loss, avg_recon_loss, avg_kl_loss
        else:
            print("No test samples available for evaluation.")
            return None

    def compare_audio(self, original_mel, fake_mel):
        min_height = min(original_mel.shape[0], fake_mel.shape[0])
        min_width = min(original_mel.shape[1], fake_mel.shape[1])
        original_mel_clipped = original_mel[:min_height, :min_width]
        fake_mel_clipped = fake_mel[:min_height, :min_width]

        orig_flat = original_mel_clipped.flatten()
        fake_flat = fake_mel_clipped.flatten()

        mse = mean_squared_error(orig_flat, fake_flat)
        if np.all(orig_flat == orig_flat[0]) or np.all(fake_flat == fake_flat[0]):
            cosine_sim = np.nan
            correlation = np.nan
        else:
            cosine_sim = 1 - cosine(orig_flat, fake_flat)
            correlation = np.corrcoef(orig_flat, fake_flat)[0, 1]

        print("=== Audio Comparison Results ===")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Cosine Similarity: {cosine_sim:.4f}")
        print(f"Correlation: {correlation:.4f}")
        return {'mse': mse, 'cosine_similarity': cosine_sim, 'correlation': correlation}

    def visualize_comparison(self, original_mel, fake_mel):
        min_height = min(original_mel.shape[0], fake_mel.shape[0])
        min_width = min(original_mel.shape[1], fake_mel.shape[1])
        original_mel_clipped = original_mel[:min_height, :min_width]
        fake_mel_clipped = fake_mel[:min_height, :min_width]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        im1 = axes[0].imshow(original_mel_clipped, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Original Mel Spectrogram')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Mel Frequency')
        plt.colorbar(im1, ax=axes[0])
        im2 = axes[1].imshow(fake_mel_clipped, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Generated Deepfake Mel Spectrogram')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Mel Frequency')
        plt.colorbar(im2, ax=axes[1])
        diff = np.abs(original_mel_clipped - fake_mel_clipped)
        im3 = axes[2].imshow(diff, aspect='auto', origin='lower', cmap='hot')
        axes[2].set_title('Absolute Difference')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Mel Frequency')
        plt.colorbar(im3, ax=axes[2])
        plt.tight_layout()
        plt.show()

    def save_audio(self, mel_spec, filename, sr=22050):
        audio = self.processor.mel_spectrogram_to_audio(mel_spec)
        sf.write(filename, audio, sr)
        print(f"Audio saved as: {filename}")
```

## Main Function
Orchestrates the workflow for data loading, training, and generation.
```python
def get_image_files(directory):
    image_files = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return image_files

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path).convert('L')
                    img_array = np.array(img, dtype=np.float32)
                    image_files.append((file_path, img_array))
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
    return image_files
```
```python
def main():
    torch.cuda.empty_cache()
    model_types = ["bsrbfkan-gelu-gated-vae", "bsrbfkan-vae", "chebyshevkanlinear-gelu-gated-vae", "chebyshevkanlinear-vae", "gelu-gated-vae", "vae-simple", "wavkan-gelu-gated-vae", "wavkan-vae"]
    
    for model_type in model_types:
        print(f"\nRunning {model_type} pipeline...")
        generator = AudioDeepfakeGenerator(height=128, width=128, model_type=model_type)
        base_path = "/kaggle/input/vae128-ctrsdd"
        train_real_path = os.path.join(base_path, "spec_128")
        
        train_files = get_image_files(train_real_path)
        if not train_files:
            print("No image files found in the training directory.")
            continue

        train_files, test_files = train_test_split(train_files, test_size=0.2, random_state=42)
        val_files = test_files

        print(f"Found {len(train_files)} training files")
        print(f"Found {len(val_files)} validation files")
        print(f"Found {len(test_files)} testing files")

        print("\nStep 1: Processing training mel spectrograms...")
        try:
            train_spectrograms = generator.preprocess_mel_spectrograms(train_files)
        except ValueError as e:
            print(f"Error: {e}")
            continue

        print("\nStep 2: Processing validation mel spectrograms...")
        try:
            val_spectrograms = generator.preprocess_mel_spectrograms(val_files)
        except ValueError as e:
            print(f"Error: {e}")
            continue

        print("\nStep 3: Processing testing mel spectrograms...")
        try:
            test_spectrograms = generator.preprocess_mel_spectrograms(test_files)
        except ValueError as e:
            print(f"Error: {e}")
            continue

        print(f"Training samples: {len(train_spectrograms)}")
        print(f"Validation samples: {len(val_spectrograms)}")
        print(f"Testing samples: {len(test_spectrograms)}")

        print("\nStep 4: Training VAE...")
        generator.train_vae(
            mel_spectrograms=train_spectrograms,
            epochs=100,
            batch_size=32,
            learning_rate=5e-5,
            beta=0.02,
            early_stop_patience=50,
            accum_steps=32
        )

        print("\nStep 5: Evaluating model on validation set...")
        val_metrics = generator.evaluate_model(val_spectrograms, beta=0.02)
        if val_metrics:
            print(f"Validation Metrics: Total Loss: {val_metrics[0]:.4f}, Recon Loss: {val_metrics[1]:.4f}, KL Loss: {val_metrics[2]:.4f}")

        print("\nStep 6: Evaluating model on test set...")
        test_metrics = generator.evaluate_model(test_spectrograms, beta=0.02)
        if test_metrics:
            print(f"Test Metrics: Total Loss: {test_metrics[0]:.4f}, Recon Loss: {test_metrics[1]:.4f}, KL Loss: {test_metrics[2]:.4f}")

        for j in range(11):
            print("\nStep 7: Preparing reference for deepfake generation...")
            reference_file, reference_mel = test_files[j]
            if reference_mel.shape != generator.target_shape:
                if reference_mel.shape[0] > generator.target_shape[0]:
                    reference_mel = reference_mel[:generator.target_shape[0], :]
                elif reference_mel.shape[0] < generator.target_shape[0]:
                    pad_height = generator.target_shape[0] - reference_mel.shape[0]
                    reference_mel = np.pad(reference_mel, ((0, pad_height), (0, 0)), 'constant', constant_values=reference_mel.min())
        
                if reference_mel.shape[1] > generator.target_shape[1]:
                    reference_mel = reference_mel[:, :generator.target_shape[1]]
                elif reference_mel.shape[1] < generator.target_shape[1]:
                    pad_width = generator.target_shape[1] - reference_mel.shape[1]
                    reference_mel = np.pad(reference_mel, ((0, 0), (0, pad_width)), 'constant', constant_values=reference_mel.min())
        
            print(f"Reference file: {os.path.basename(reference_file)}, shape: {reference_mel.shape}")
            generator.processor.plot_mel_spectrogram(reference_mel, "Original Reference Mel Spectrogram")
        
            print("\nStep 8: Generating high-quality deepfakes...")
            noise_levels = [0.0]
            deepfake_results = generator.generate_deepfake(reference_mel, noise_levels=noise_levels)
        
            for i, (fake_mel, noise_level) in enumerate(deepfake_results):
                print(f"Generated deepfake variation {j+1}/{len(deepfake_results)} (noise={noise_level})...")
                generator.save_audio(fake_mel, f"high_quality_deepfake_{j+1}_noise_{noise_level}.wav")
                results = generator.compare_audio(reference_mel, fake_mel)
                print(f"  MSE: {results['mse']:.4f}, Cosine Sim: {results['cosine_similarity']:.4f}, Correlation: {results['correlation']:.4f}")
        
            best_idx = 0
            best_fake_mel, best_noise = deepfake_results[best_idx]
        
            print(f"\nStep 9: Detailed analysis of best deepfake (noise={best_noise})...")
            detailed_results = generator.compare_audio(reference_mel, best_fake_mel)
        
            print("\nStep 10: Visualizing results...")
            generator.visualize_comparison(reference_mel, best_fake_mel)
            generator.save_audio(reference_mel, "original_reference_improved.wav")
        
            print("\nDeepfake generation completed for {model_type}!")
            print(f"Generated {len(deepfake_results)} high-quality variations")
            print("Files saved:")
            print("- original_reference_improved.wav")
            for i, (_, noise) in enumerate(deepfake_results):
                print(f"- high_quality_deepfake_{i+1}_noise_{noise}.wav")

            print(f"\nPerformance Summary for {model_type}:")
            print(f"Best deepfake metrics:")
            print(f"  MSE: {detailed_results['mse']:.6f}")
            print(f"  Cosine Similarity: {detailed_results['cosine_similarity']:.6f}")
            print(f"  Correlation: {detailed_results['correlation']:.6f}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
```

## Project Structure

- RadialBasisFunction: Implements radial basis functions for BSRBFKANLinear.
- BSRBFKANLinear: Combines B-splines and radial basis functions for enhanced linear transformations.
- GeluGatedLayer: Provides a GELU-activated gated layer for VAE enhancements.
- ChebyshevKANLinear: Utilizes Chebyshev polynomials for advanced transformations with normalization.
- WavKANLinear: Implements a wavelet-based KAN linear layer with chunked processing.
- ImprovedAudioVAE: Base VAE architecture with support for custom layers.
- AudioProcessor: Handles Mel spectrogram conversion and visualization.
- AudioDataset: PyTorch Dataset for loading Mel spectrograms.
- AudioDeepfakeGenerator: Main class orchestrating preprocessing, model training, deepfake generation, evaluation, and visualization.
- Main Function: Manages the workflow, including data loading, splitting, and execution of the pipeline for all VAE variants.


## Training Configuration
- **Epochs**: 100 (with early stopping after 50 epochs of no improvement).
- **Batch Size**: 32
- **Learning Rate**: 5e-5
- **KL Divergence Weight (β)**: 0.02
- **Gradient Accumulation Steps**: 32
- **Latent Dimension**: 256
- **Dropout Rate**: 0.3

## Evaluation Metrics
For each generated deepfake:
- **Mean Squared Error (MSE)**: Measures pixel-wise differences.
- **Cosine Similarity**: Assesses structural similarity.
- **Pearson Correlation**: Evaluates linear correlation between spectrograms.

## Example Output
```plaintext
Step 1:iksi Processing training mel spectrograms...
Successfully processed 1000 files
Final data shape: (1000, 1, 128, 128)

Step 4: Training convolutional VAE...
Epoch 1/100: Loss: 0.1234 | Recon: 0.1000 | KL: 0.0234
Best VAE model saved.

Step 8: Generating high-quality deepfakes...
Generated deepfake variation 1/1 (noise=0.0)...
  MSE: 0.0123, Cosine Sim: 0.9876, Correlation: 0.9901

Files saved:
- original_reference_improved.wav
- high_quality_deepfake_1_noise_0.0.wav
```

## Notes
- Ensure sufficient GPU memory for training (CUDA out-of-memory errors are mitigated with `torch.cuda.empty_cache()` and gradient accumulation).
- The dataset directory (`/kaggle/input/vae128-ctrsdd/spec_128`) must contain valid spectrogram images.
- Adjust `height`, `width`, and other hyperparameters in `AudioDeepfakeGenerator` for different dataset requirements.
- The code includes a reference to `KANLinear`, which is not defined in the provided code. Ensure compatibility or remove this reference if not using a custom KANLinear layer.

## Acknowledgments
- Built with PyTorch, librosa, and other open-source libraries.
- Inspired by deep learning techniques for audio synthesis and deepfake generation.
