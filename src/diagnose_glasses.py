"""
BCG -- glasses connection diagnostic.

Run this WHILE connected to the ESP32-CAM WiFi. It checks, step by step,
exactly where the connection to the glasses camera breaks. No internet needed.

IMPORTANT: close any browser tab showing the stream BEFORE running this --
the ESP32 usually allows only ONE stream client at a time.

Run:  python src/diagnose_glasses.py
"""

import socket
import subprocess

HOST = "192.168.4.1"
PORT = 8080


def line(msg):
    print(msg)


line("\n================ GLASSES CONNECTION DIAGNOSTIC ================\n")

# --- 1. Which WiFi are we on? -------------------------------------------------
ssid = None
try:
    out = subprocess.check_output(
        ["netsh", "wlan", "show", "interfaces"], text=True, errors="ignore"
    )
    for ln in out.splitlines():
        s = ln.strip()
        if s.startswith("SSID") and "BSSID" not in s:
            ssid = s.split(":", 1)[1].strip()
        if s.startswith("IPv4") or "IP address" in s:
            pass
except Exception as e:
    line(f"[1] Could not read WiFi info: {e}")

line(f"[1] Connected WiFi (SSID): {ssid}")
if ssid and "ESP32" not in ssid:
    line("    -> WRONG WiFi. Connect to ESP32-CAM first, then re-run.\n")
    raise SystemExit(0)
else:
    line("    -> OK, on the glasses WiFi.\n")

# --- 2. What IP did the laptop get? ------------------------------------------
my_ip = None
try:
    out = subprocess.check_output(["ipconfig"], text=True, errors="ignore")
    block = []
    for ln in out.splitlines():
        if "IPv4" in ln:
            block.append(ln.split(":", 1)[1].strip())
    # pick a 192.168.4.x address if present
    for ip in block:
        if ip.startswith("192.168.4."):
            my_ip = ip
    if not my_ip and block:
        my_ip = block[-1]
except Exception as e:
    line(f"[2] Could not read IP: {e}")

line(f"[2] Laptop IP on this network: {my_ip}")
if my_ip and my_ip.startswith("192.168.4."):
    line("    -> OK, got an address from the glasses (192.168.4.x).\n")
elif my_ip and my_ip.startswith("169.254."):
    line("    -> BAD: 169.254.x.x means the glasses did NOT give an IP (DHCP failed).")
    line("       Disconnect/reconnect the ESP32-CAM WiFi and try again.\n")
else:
    line("    -> Not a 192.168.4.x address -- may be on the wrong network.\n")

# --- 3. Can we open a raw TCP connection to the camera? ----------------------
line(f"[3] Trying raw TCP connect to {HOST}:{PORT} ...")
s = socket.socket()
s.settimeout(6)
connected = False
try:
    s.connect((HOST, PORT))
    connected = True
    line("    -> OK, TCP connection opened.\n")
except Exception as e:
    line(f"    -> FAILED: {e}")
    line("       The camera port is not reachable. Most likely causes:")
    line("       a) a browser tab is still holding the only stream connection -> close it")
    line("       b) the ESP32 web server isn't running / wrong port\n")
    s.close()
    raise SystemExit(0)

# --- 4. Ask for the stream and look at the response header -------------------
line("[4] Sending HTTP request and reading the response header ...")
try:
    s.sendall(f"GET / HTTP/1.0\r\nHost: {HOST}\r\n\r\n".encode())
    data = s.recv(500)
    text = data.decode("latin-1", "replace")
    line("    --- first bytes of response ---")
    for ln in text.splitlines()[:12]:
        line("    " + ln)
    line("    -------------------------------")
    low = text.lower()
    if "multipart/x-mixed-replace" in low or "image/jpeg" in low:
        line("\n    -> This IS an MJPEG stream. The live_detection.py reader should work.")
    elif "text/html" in low:
        line("\n    -> This is an HTML page, not the raw stream.")
        line("       The actual stream may be at a sub-path (e.g. /stream).")
    else:
        line("\n    -> Got a response but couldn't identify the type (see bytes above).")
except Exception as e:
    line(f"    -> Could connect but reading failed: {e}")
finally:
    s.close()

line("\n================ END OF DIAGNOSTIC ================\n")
