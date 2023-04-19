#Send message example

import socket

# set up socket connection
HOST = "192.168.8.50"  # IP address of turtlebot robot
PORT = 3020  # port number for socket communication
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))


message = "left".encode()
sock.send(message)
sock.close()