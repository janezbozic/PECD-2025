import time
import board
import adafruit_dht

# Sensor data pin is connected to GPIO 4
# sensor = adafruit_dht.DHT22(board.D4)
sensor = adafruit_dht.DHT11(board.D4)

while True:
    try:
        # Print the values to the serial port
        temperature_c = sensor.temperature
        humidity = sensor.humidity
        print("Temp={0:0.1f}ºC, Humidity={1:0.1f}%".format(temperature_c, humidity))

    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        print(error.args[0])
        time.sleep(2.0)
        continue
    except Exception as error:
        sensor.exit()
        raise error

    time.sleep(1.0)

