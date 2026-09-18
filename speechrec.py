import os
import sys
import glob
import time
import json
import numpy as np
import sounddevice as sd

# Audio processing constants
SAMPLE_RATE = 16000
DURATION = 4
N_FFT = 1024  # Yields (1024 // 2) + 1 = 513 features
TRAIN_DIR = "train"
REF_FILE = os.path.join(TRAIN_DIR, "refdata.json")

os.makedirs(TRAIN_DIR, exist_ok=True)


def capture_averaged_spectrum(duration=DURATION, samplerate=SAMPLE_RATE, n_fft=N_FFT):
    """Captures audio and computes an averaged 513-length FFT magnitude spectrum."""
    print(f"\nRecording for {duration} seconds... Speak now!")
    audio = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32"
    )
    sd.wait()
    print("Recording finished. Processing audio spectrum...")

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
    # Log compression for stable numeric scale
    mean_spectrum = np.log1p(mean_spectrum)
    return mean_spectrum.tolist()


def rec_ref():
    """Records the reference audio spectrum and saves it to train/refdata.json."""
    spectrum = capture_averaged_spectrum()
    with open(REF_FILE, "w") as f:
        json.dump(spectrum, f, indent=4)
    print(f"Reference voiceprint saved -> {REF_FILE} ({len(spectrum)} bins)")


def rec_test_samp():
    """Records a labeled training/test sample and saves it with the 0_out or 1_out suffix."""
    label = input(
        "Enter sample label (1 = you, 0 = someone else / background): "
    ).strip()
    while label not in ("0", "1"):
        label = input("Invalid input. Please enter 0 or 1: ").strip()

    spectrum = capture_averaged_spectrum()
    timestamp = int(time.time() * 1000)
    filename = os.path.join(TRAIN_DIR, f"sample_{timestamp}_{label}_out.json")

    with open(filename, "w") as f:
        json.dump(spectrum, f, indent=4)
    print(f"Sample saved -> {filename}")


def rem_test_samps():
    """Deletes all generated *_out.json samples in the train directory."""
    files = glob.glob(os.path.join(TRAIN_DIR, "*_out.json"))
    if not files:
        print(f"No sample files found in '{TRAIN_DIR}'.")
        return

    confirm = (
        input(f"Are you sure you want to delete {len(files)} sample file(s)? (y/n): ")
        .strip()
        .lower()
    )
    if confirm == "y":
        for file in files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error removing {file}: {e}")
        print(f"Deleted {len(files)} sample files.")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    # If passed as command-line arguments (e.g.: python audio_recorder.py recref)
    if len(sys.argv) > 1:
        flag = sys.argv[1].lower().replace("-", "").replace("_", "")
        if flag == "recref":
            rec_ref()
        elif flag == "rectestsamp":
            rec_test_samp()
        elif flag == "remtestsamps":
            rem_test_samps()
        else:
            print(f"Unknown flag: '{sys.argv[1]}'")
            print("Available flags: recref, rectestsamp, remtestsamps")
    else:
        # Interactive loop
        while True:
            print("\n--- Audio Recorder Menu ---")
            print("1. recref       (Record reference audio)")
            print("2. rectestsamp  (Record new 0_out / 1_out sample)")
            print("3. remtestsamps (Delete all recorded sample files)")
            print("4. exit")
            choice = (
                input("Select option or enter flag: ")
                .strip()
                .lower()
                .replace("-", "")
                .replace("_", "")
            )

            if choice in ("1", "recref"):
                rec_ref()
            elif choice in ("2", "rectestsamp"):
                rec_test_samp()
            elif choice in ("3", "remtestsamps"):
                rem_test_samps()
            elif choice in ("4", "exit", "q"):
                break
            else:
                print("Invalid command.")
