from gpiozero import MCP3008
from time import sleep

#create an object called pot that refers to MCP3008 channel 0
pot = MCP3008(0)

while(True):
    print(pot.value)
    sleep(1)
