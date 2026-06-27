# Install & run (Art-Net home rig)

Music-reactive lighting over Art-Net, driven by the audio coming off the mixer.
No USB-DMX hardware needed — just an Art-Net node on the network.

## 1. Install uv (one-liner)

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Open a new terminal afterwards so `uv` is on your PATH.

## 2. Get the code & install dependencies

```bash
git clone https://github.com/YinonLensor/Oculizer.git
cd Oculizer
uv sync
```
`uv sync` reads `pyproject.toml` + `uv.lock` and builds an isolated `.venv` with
the exact pinned versions. First run downloads a few packages; after that it's
instant. (Python is handled by uv — you don't need to install it yourself.)

## 3. Run

```bash
uv run run_artnet.py
```

That's it. The Art-Net node IP defaults to **192.168.2.15**, universe 0.
If your node is at a different address:
```bash
OCULIZER_ARTNET_IP=192.168.x.y uv run run_artnet.py
```

### Controls
While it's running, type a key + **Enter**:

| Key | Scene |
|-----|-------|
| `1` | home_ambient — chill blue wash |
| `2` | home_party — bass-reactive, lasers, strobe on drops |
| `3` | home_laser — lasers forward |
| `4` | home_strobe — full strobe |
| `r` | reload scenes from disk (edit the JSON, see it live) |
| `q` | quit (sends blackout) |

## Hardware checklist

- **Art-Net node** reachable on the network at the IP above (universe 0).
- **DMX patch** (addresses must match — Oculizer assigns them in order):
  - Party light → DMX address **1** (uses channels 1–7)
  - Disco-ball pinspot (9-channel mode) → DMX address **8** (uses 8–16)
- **Audio input**: defaults to the **Allen & Heath Xone:24C** (its USB soundcard
  streams the master mix back to the computer in STREAM mode). Using a different
  interface? Edit `input_device="xone"` near the bottom of `run_artnet.py` to a
  substring of your device's name.

## Troubleshooting

- **No lights**: confirm the node IP, that the computer is on the same subnet,
  and that fixtures are addressed as above.
- **No reaction to music**: make sure the Xone (or your interface) is the input
  and the master mix is actually reaching the computer.
- **Want the ML auto-scene engine** (optional, heavy): `uv run --extra prediction
  run_artnet.py`, plus install EfficientAT as noted in `pyproject.toml`.

---
Forked from [LandryBulls/Oculizer](https://github.com/LandryBulls/Oculizer) (MIT).
This fork adds Art-Net output, Xone:24C input, and custom party-light / disco-ball
fixtures. See `FIXTURES`-style channel notes in `oculizer/light/mapping.py`.
