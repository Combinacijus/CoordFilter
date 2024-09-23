
import time
import winsound
import random

def beep_every_x_seconds(t, ms):
        winsound.Beep(1000 + int(random.random()*800-400), ms)
        time.sleep(t)

beep_every_x_seconds(0.2, 750)
while True:
    for i in range(5):
        print(f"Running {i+1}")
        beep_every_x_seconds(0.1, 700)
        beep_every_x_seconds(25, 1000)
        print(f"Waiting {i+1}")
    beep_every_x_seconds(30, 2000)
    