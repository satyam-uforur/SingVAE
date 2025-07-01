# Audio Deepfake Generator using Variational Autoencoder (VAE)

This project implements an end-to-end pipeline for generating deepfake audio using a Variational Autoencoder (VAE) trained on Mel spectrograms. The system processes audio data, trains a convolutional VAE, generates deepfake variations, and evaluates the results with visualizations and metrics.

## Features

- **Audio Preprocessing**: Converts audio files to Mel spectrograms with configurable parameters (sample rate, n_mels, n_fft, hop_length).
- **VAE Model**: A convolutional VAE with customizable architecture (filters, kernels, strides, latent dimension) for learning audio representations.
- **Deepfake Generation**: Generates high-quality deepfake audio by adding controlled noise in the latent space.
- **Evaluation Metrics**: Computes Mean Squared Error (MSE), Cosine Similarity, and Pearson Correlation between original and generated spectrograms.
- **Visualization**: Displays Mel spectrograms, training loss curves, and comparison plots for original vs. generated audio.
- **Audio Output**: Saves generated and original audio as WAV files using librosa and soundfile.

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

## Code Snippets

### AudioProcessor Class
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
        librosa.display.specshow(
            mel_spec_db, sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()
```

### ImprovedAudioVAE Class
Defines the convolutional VAE model.

```python
class ImprovedAudioVAE(nn.Module):
    def __init__(self, input_shape, conv_filters=(16, 32, 64, 128), conv_kernels=(3, 3, 3, 3),
                 conv_strides=(1, 2, 2, 2), latent_dim=256, dropout_rate=0.3):
        super(ImprovedAudioVAE, self).__init__()
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
服从
        self.num_conv_layers = len(conv_filters)
        self.shape_before_bottleneck = None

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.apply(self._init_weights)

    def _build_encoder(self):
        layers = []
        in_channels = 1
        for i, (filters, kernel, stride) in enumerate(zip(self.conv_filters, self.conv_kernels, self.conv_strides)):
            layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(filters),
                nn.Dropout(self.dropout_rate)
            ])
            in_channels = filters
        # ... (additional encoder and decoder logic)
```

### Main Function
Orchestrates the workflow for data loading, training, and generation.

```python
def main():
    torch.cuda.empty_cache()
    generator = AudioDeepfakeGenerator(height=128, width=128)
    base_path = "/kaggle/input/vae128-ctrsdd"
    train_real_path = os.path.join(base_path, "spec_128")
    
    train_files = get_image_files(train_real_path)
    if not train_files:
        print("No image files found in the training directory.")
        return

    train_files, test_files = train_test_split(train_files, test_size=0.2, random_state=42)
    val_files = test_files

    print(f"Found {len(train_files)} training files")
    print(f"Found {len(val_files)} validation files")
    print(f"Found {len(test_files)} testing files")

    print("\nStep 1: Processing training mel spectrograms...")
    train_spectrograms = generator.preprocess_mel_spectrograms(train_files)
    # ... (additional processing and training logic)
```

## Project Structure

- **AudioProcessor**: Handles Mel spectrogram conversion and visualization.
- **AudioDataset**: PyTorch Dataset for loading Mel spectrograms.
- **ImprovedAudioVAE**: Convolutional VAE model with encoder, bottleneck, and decoder.
- **AudioDeepfakeGenerator**: Main class orchestrating preprocessing, VAE training, deepfake generation, evaluation, and visualization.
- **Main Function**: Manages the workflow, including data loading, splitting, and execution of the pipeline.

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

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with PyTorch, librosa, and other open-source libraries.
- Inspired by deep learning techniques for audio synthesis and deepfake generation.