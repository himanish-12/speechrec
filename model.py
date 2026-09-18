import os
import glob
import json
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Audio constants matching your recorder
SAMPLE_RATE = 16000
DURATION = 4
N_FFT = 1024  # (1024 // 2) + 1 = 513 features

# Storage paths
TRAIN_DIR = "train"
REF_FILE = os.path.join(TRAIN_DIR, "refdata.json")
WEIGHTS_FILE = "model_weights.json"

# Device selection: Intel Arc GPU (XPU) prioritized over CPU
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")


# 1. ArcNet Architecture (~2M Parameters)
class ArcNet(nn.Module):
    def __init__(self, in_features=513):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 432),
            nn.BatchNorm1d(432),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(432, 1),  # Raw unconstrained logit
        )

    def forward(self, x):
        return self.net(x)


# 2. Live Audio Feature Extractor
def capture_live_spectrum(duration=DURATION, samplerate=SAMPLE_RATE, n_fft=N_FFT):
    """Records live audio and computes an averaged 513-feature FFT spectrum."""
    print(f"\n[LIVE TEST] Recording for {duration} seconds... Speak now!")
    audio = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32"
    )
    sd.wait()
    print("Recording finished. Processing live voiceprint...")

    signal = audio.flatten()
    hop_length = n_fft // 2
    window = np.hamming(n_fft)

    spectra = []
    for start in range(0, len(signal) - n_fft, hop_length):
        frame = signal[start : start + n_fft] * window
        mag = np.abs(np.fft.rfft(frame, n=n_fft))
        spectra.append(mag)

    if not spectra:
        raise RuntimeError("Recording was too short to compute spectral frames.")

    mean_spectrum = np.mean(spectra, axis=0)
    # Log compression matching audio_recorder.py
    mean_spectrum = np.log1p(mean_spectrum)
    return mean_spectrum.tolist()


# 3. JSON Weight Serialization Helpers
def save_model_to_json(model, filepath=WEIGHTS_FILE):
    state = model.state_dict()
    serializable_dict = {
        name: tensor.detach().cpu().tolist() for name, tensor in state.items()
    }
    with open(filepath, "w") as f:
        json.dump(serializable_dict, f)
    print(f"Model parameters saved to -> '{filepath}'")


def load_model_from_json(model, filepath=WEIGHTS_FILE):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"'{filepath}' not found. Please train the model first."
        )

    with open(filepath, "r") as f:
        serializable_dict = json.load(f)

    current_state = model.state_dict()
    restored_state = {}
    for name, tensor in current_state.items():
        if name in serializable_dict:
            restored_state[name] = torch.tensor(
                serializable_dict[name], dtype=tensor.dtype, device=device
            )
        else:
            restored_state[name] = tensor

    model.load_state_dict(restored_state)
    print(f"Model parameters loaded from '{filepath}'")


# 4. Dataset for Training
class SpeechDiffDataset(Dataset):
    def __init__(self, sample_files, ref_path=REF_FILE):
        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Reference file '{ref_path}' missing. Run recref first."
            )

        with open(ref_path, "r") as f:
            self.ref_tensor = torch.tensor(json.load(f), dtype=torch.float32)

        self.samples = []
        for path in sample_files:
            filename = os.path.basename(path).lower()
            if "0_out" in filename:
                label = 0.0
            elif "1_out" in filename:
                label = 1.0
            else:
                continue

            with open(path, "r") as f:
                sample_tensor = torch.tensor(json.load(f), dtype=torch.float32)

            # Subtraction: (Sample - Ref)
            diff = sample_tensor - self.ref_tensor
            self.samples.append((diff, torch.tensor([label], dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# 5. Training Routine
def train_routine():
    if not os.path.exists(REF_FILE):
        print(f"Cannot train: Missing '{REF_FILE}'. Record reference first.")
        return

    sample_files = glob.glob(os.path.join(TRAIN_DIR, "*_out.json"))
    if not sample_files:
        print(f"No sample files found in '{TRAIN_DIR}'. Record samples first.")
        return

    print(f"Loading {len(sample_files)} samples for training on device: {device}...")
    dataset = SpeechDiffDataset(sample_files, ref_path=REF_FILE)

    batch_size = min(32, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ArcNet(in_features=513).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    model.train()
    epochs = 30
    for epoch in range(epochs):
        running_loss = 0.0
        for diffs, labels in loader:
            diffs, labels = diffs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(diffs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * diffs.size(0)

        epoch_loss = running_loss / len(dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}")

    save_model_to_json(model, WEIGHTS_FILE)


# 6. Automatic Live Test Routine
def live_test_routine():
    if not os.path.exists(REF_FILE):
        print(f"Cannot test: Missing '{REF_FILE}'. Record your reference voice first.")
        return

    if not os.path.exists(WEIGHTS_FILE):
        print(f"Cannot test: Missing '{WEIGHTS_FILE}'. Train the model first.")
        return

    # Load reference vector from disk
    with open(REF_FILE, "r") as fr:
        ref_tensor = torch.tensor(json.load(fr), dtype=torch.float32)

    # 1. Trigger live recording automatically
    live_sample = capture_live_spectrum()
    sample_tensor = torch.tensor(live_sample, dtype=torch.float32)

    # 2. Compute difference (Live Audio - Reference) and move to Arc GPU
    diff = (sample_tensor - ref_tensor).unsqueeze(0).to(device)

    # 3. Load model and run inference
    model = ArcNet(in_features=513).to(device)
    load_model_from_json(model, WEIGHTS_FILE)
    model.eval()

    with torch.no_grad():
        logit = model(diff)
        prob = torch.sigmoid(logit).item()

    pred_class = 1 if prob >= 0.5 else 0
    confidence = prob if pred_class == 1 else (1.0 - prob)

    # 4. Display result
    print("\n" + "=" * 45)
    print("               LIVE INFERENCE RESULT")
    print("=" * 45)
    print(
        f"Verification Result: {'MATCH (You - Class 1)' if pred_class == 1 else 'MISMATCH (Not You - Class 0)'}"
    )
    print(f"Predicted Class:     {pred_class}")
    print(f"Confidence:          {confidence * 100:.2f}%")
    print(f"Probability Score:   {prob:.4f}")
    print("=" * 45)


if __name__ == "__main__":
    while True:
        action = input("\nChoose task: [train], [test], or [exit]: ").strip().lower()
        if action in ("train", "tr"):
            train_routine()
        elif action in ("test", "te"):
            live_test_routine()
        elif action in ("exit", "quit", "q"):
            break
        else:
            print("Please choose 'train', 'test', or 'exit'.")
