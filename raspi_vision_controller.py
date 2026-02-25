#!/usr/bin/env python3
"""
Simple PC Vision Controller
SPACE key → Capture image
Q or Ctrl+C → Exit
"""

import socket
import json
import struct
from datetime import datetime
from pathlib import Path
import sys
import termios
import tty


def get_key():
    """Read single keypress (Linux only)"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


class SimplePCController:
    def __init__(self, pi_ip='192.168.50.3', pi_port=5000, save_folder="watch_folder"):
        self.pi_ip = pi_ip
        self.pi_port = pi_port
        self.socket = None
        self.connected = False

        self.save_folder = Path(save_folder)
        self.save_folder.mkdir(parents=True, exist_ok=True)
        print(f"📁 Saving images to: {self.save_folder.resolve()}")

    def connect(self):
        try:
            print(f"📡 Connecting to {self.pi_ip}:{self.pi_port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.pi_ip, self.pi_port))
            self.socket.settimeout(None)

            self._receive_response()
            self.connected = True
            print("✅ Connected")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def disconnect(self):
        if self.socket:
            self.socket.close()
        print("👋 Disconnected")

    def _send_command(self, command):
        data = json.dumps(command).encode("utf-8")
        msg = struct.pack(">I", len(data)) + data
        self.socket.sendall(msg)

    def _recvall(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def _receive_response(self):
        raw_len = self._recvall(4)
        if not raw_len:
            return None
        msglen = struct.unpack(">I", raw_len)[0]
        return json.loads(self._recvall(msglen).decode())

    def _receive_image(self):
        raw_len = self._recvall(4)
        imglen = struct.unpack(">I", raw_len)[0]
        print(f"📥 Receiving image ({imglen:,} bytes)")
        return self._recvall(imglen)

    def capture_image(self):
        if not self.connected:
            print("❌ Not connected")
            return

        print("📸 Capture triggered")

        self._send_command({"action": "capture"})
        response = self._receive_response()
        if response.get("status") != "success":
            print("❌ Capture failed")
            return

        img_data = self._receive_image()
        if not img_data:
            print("❌ Image receive failed")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.save_folder / f"img_{ts}.jpg"
        with open(path, "wb") as f:
            f.write(img_data)

        print(f"✅ Saved: {path}")

    def run(self):
        print("\n==============================")
        print(" SPACE  → Capture Image")
        print(" Q      → Quit")
        print("==============================\n")

        if not self.connect():
            return

        try:
            while True:
                key = get_key()

                if key == " ":
                    self.capture_image()

                elif key.lower() == "q":
                    break

        except KeyboardInterrupt:
            pass

        self.disconnect()


def main():
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.50.3"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    folder = sys.argv[3] if len(sys.argv) > 3 else "watch_folder"

    SimplePCController(ip, port, folder).run()


if __name__ == "__main__":
    main()

