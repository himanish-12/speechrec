import json
import numpy as np
import sounddevice as sd

file = open("spdata.json", "w")
DTFT2 = []

try:
    ind = 0
    with sd.InputStream(samplerate=10240, channels=1, blocksize=1024) as stream:
        while True:
            ind += 1
            chunnk, overflowed = stream.read(1024)
            if overflowed:
                print("overflowing data")
            sig = chunnk[:, 0] * np.hamming(1024)
            DTFT1 = np.fft.rfft(sig)
            DTFT2 = np.abs(DTFT1).tolist()
            print(DTFT2)

except KeyboardInterrupt:
    print("\nRecording stopped by user.")

finally:
    # FINALLY guarantees this runs no matter what happens
    json.dump(DTFT2, file, indent=4)
    file.flush()
    file.close()
    print("JSON file successfully written and closed!")
