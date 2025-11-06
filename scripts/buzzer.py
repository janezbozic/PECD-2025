import RPi.GPIO as GPIO
import time

BUZZER_PIN = 18  # BCM numbering
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)

def beep(on=0.2, off=0.2, times=3):
    for _ in range(times):
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(on)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(off)

try:
    print("3 short beeps:")
    beep(1.0, 1.0, 3)

    print("1 long beep:")
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(5.0)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

finally:
    GPIO.cleanup()

