import numpy as np
import sounddevice as sd
import json
import time


def check():
    try:
        ind = 0
        DTFT = []
        start = time.time()
        with sd.InputStream(samplerate=1000, channels=1, blocksize=1000) as stream:
            while True:
                ind += 1
                if time.time() - start > 4:
                    with open("recdata.json", "w") as file:
                        json.dump(DTFT, file, indent=4)
                    print("ending")
                    break
                chunnk, overflowed = stream.read(1000)
                print(f"listening index{ind}")
                if overflowed:
                    print("overflowing data")
                sig = chunnk[:, 0] * np.hamming(1000)
                DTFT = np.abs(np.fft.rfft(sig)).tolist()

    except KeyboardInterrupt:
        print("keyinterrupt")
    print("\nRecording stopped by user.")


def record():
    DTFT2 = []
    try:
        ind = 0
        start = time.time()
        with sd.InputStream(samplerate=1000, channels=1, blocksize=1000) as stream:
            while True:
                ind += 1
                chunnk, overflowed = stream.read(1000)
                if time.time() - start > 4:
                    print("ending")
                    break

                # print(f"listening index{ind}")
                if overflowed:
                    print("overflowing data")
                sig = chunnk[:, 0] * np.hamming(1000)
                DTFT1 = np.fft.rfft(sig)
                DTFT2 = np.abs(DTFT1).tolist()
                print(DTFT2)
    except KeyboardInterrupt:
        print("keyinterrupt")
    with open("spdata.json", "w") as file:
        json.dump(DTFT2, file, indent=4)


while True:
    choise = input("chose mode : ")
    if choise == "r":
        record()
    elif choise == "c":
        check()
    else:
        print("please chose a valid option")
