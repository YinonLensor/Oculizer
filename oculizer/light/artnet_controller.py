"""ArtnetProController — a drop-in replacement for EnttecProController that
emits Art-Net/UDP instead of Enttec USB serial.

It exposes the same interface Oculizer's fixtures rely on:
    - self.dmx_data          (513-byte list; index 0 = start code, 1..512 = channels)
    - _send_dmx_packet()     (push the current frame)
    - send_dmx(), set_channel(), set_channels(), blackout(), close()

so control.py can swap controllers without touching any fixture code.

Unlike the serial controller, a background thread re-sends the current frame at
a steady rate (default 30 Hz). Many Art-Net nodes blank their output if frames
stop arriving, so this keep-alive prevents flicker between Oculizer's
event-driven updates.
"""

import socket
import struct
import threading
import time
from typing import List


class ArtnetProController:
    def __init__(self, node_ip: str, universe: int = 0, port: int = 6454,
                 refresh_hz: int = 30, **_ignored):
        self.node_ip = node_ip
        self.universe = universe
        self.port = port
        self.dmx_data = [0] * 513  # index 0 = DMX start code (not sent), 1..512 = channels
        self._seq = 0
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._running = True
        self._thread = threading.Thread(
            target=self._keepalive, args=(refresh_hz,), daemon=True
        )
        self._thread.start()
        print(f"Art-Net controller -> {node_ip} (universe {universe}) @ {refresh_hz} Hz keep-alive")

    def _build_packet(self) -> bytes:
        self._seq = (self._seq % 255) + 1
        body = bytes(self.dmx_data[1:513])  # 512 channels, drop the start code
        return (
            b"Art-Net\x00"
            + struct.pack("<H", 0x5000)              # OpDmx
            + struct.pack(">H", 14)                  # protocol v14
            + bytes([self._seq, 0,
                     self.universe & 0xFF,
                     (self.universe >> 8) & 0x7F])
            + struct.pack(">H", len(body))
            + body
        )

    def _send(self):
        with self._lock:
            pkt = self._build_packet()
        self._sock.sendto(pkt, (self.node_ip, self.port))

    def _keepalive(self, hz: int):
        period = 1.0 / max(1, hz)
        while self._running:
            self._send()
            time.sleep(period)

    # --- EnttecProController-compatible surface ---------------------------------

    def _send_dmx_packet(self):
        self._send()

    def send_dmx(self, data: List[int], start_channel: int = 1):
        if not data:
            return
        with self._lock:
            for i, value in enumerate(data):
                ch = start_channel + i
                if 1 <= ch <= 512:
                    self.dmx_data[ch] = max(0, min(255, int(value)))
        self._send()

    def set_channel(self, channel: int, value: int):
        if 1 <= channel <= 512:
            with self._lock:
                self.dmx_data[channel] = max(0, min(255, int(value)))
            self._send()

    def set_channels(self, channels: List[int], values: List[int]):
        if len(channels) != len(values):
            raise ValueError("Channels and values lists must have the same length")
        with self._lock:
            for channel, value in zip(channels, values):
                if 1 <= channel <= 512:
                    self.dmx_data[channel] = max(0, min(255, int(value)))
        self._send()

    def blackout(self):
        with self._lock:
            self.dmx_data = [0] * 513
        self._send()

    def close(self):
        self._running = False
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        self.blackout()
        self._sock.close()
        print("Art-Net controller closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
