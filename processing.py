import numpy as np
import sounddevice as sd
import json

file = open("spdata.json", "w")

DTFT2 = []
try:
    ind = 0
    with sd.InputStream(samplerate=10240, channels=1, blocksize=1024) as stream:
        while True:
            ind += 1
            chunnk, overflowed = stream.read(1024)
            # print(f"listening index{ind}")
            if overflowed:
                print("overflowing data")
            sig = chunnk[:, 0] * np.hamming(1024)
            DTFT1 = np.fft.rfft(sig)
            DTFT2 = np.abs(DTFT1).tolist()
            print(DTFT2)
except KeyboardInterrupt:
    json.dump(DTFT2, file, indent=4)
    file.close()
    print("\nRecording stopped by user.")
