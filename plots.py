import json
import matplotlib.pyplot as plt
import numpy as np

# Load the reference data and the recorded speaker data
with open("spdata.json", "r") as f:
    spdata = json.load(f)

with open("recdata.json", "r") as f:
    recdata = json.load(f)

# Convert to numpy arrays
arr_sp = np.array(spdata)
arr_rec = np.array(recdata)

# Generate frequency axis (1000 Hz sample rate, 1000 block size)
freqs = np.fft.rfftfreq(1000, 1 / 1000)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(freqs, arr_sp, label="Reference (spdata)", color="blue", alpha=0.7)
plt.plot(freqs, arr_rec, label="New Speaker (recdata)", color="orange", alpha=0.7)

plt.title("Voice Spectrum Comparison (DTFT Magnitude)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)

# Save the plot to an image file instead of opening a window
plt.savefig("spectrum_comparison.png", dpi=300)
print("Plot successfully saved as spectrum_comparison.png!")
