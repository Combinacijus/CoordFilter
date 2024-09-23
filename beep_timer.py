
import time
import winsound

def beep_every_x_seconds(t, ms):
        winsound.Beep(1000, ms)
        time.sleep(t)

beep_every_x_seconds(0.2, 650)
while True:
    for i in range(5):
        print(f"Running {i+1}")
        beep_every_x_seconds(30, 650)
        print(f"Waiting {i+1}")
    beep_every_x_seconds(30, 1700)
    