#!/usr/bin/env python3
"""Run Oculizer over Art-Net with the 'yinon' profile, listening to the Xone:24C.

This renders Oculizer's scene engine (mfft/time/bool modulators + scenes) through
your real fixtures: the 7-channel party light (DMX 1-7) and the disco-ball
pinspot (DMX 8-13), all on Art-Net universe 0 -> 192.168.2.57.

Switch scenes by typing the scene's key + Enter. 'r' reloads scenes from disk
(edit the JSON and see it live). 'q' quits (sends blackout).

Run:
  cd ~/personal-dev/Oculizer
  OCULIZER_ARTNET_IP=192.168.2.57 uv run run_artnet.py

Dependencies come from pyproject.toml (numba is pinned >=0.61 so librosa doesn't
drag in the ancient numba 0.53.1 that can't build on Python 3.13). For the ML
scene-prediction path add `--extra prediction` (and install EfficientAT per the
note in pyproject.toml).
"""
import os
import sys

# Default the node IP so the controller picks Art-Net instead of USB serial.
os.environ.setdefault("OCULIZER_ARTNET_IP", "192.168.2.57")
os.environ.setdefault("OCULIZER_ARTNET_UNIVERSE", "0")

from oculizer.scenes import SceneManager
from oculizer.light.control import Oculizer

PROFILE = "yinon"
START_SCENE = "home_ambient"
OUR_SCENES = ["home_ambient", "home_party", "home_laser", "home_strobe"]


def main():
    scene_manager = SceneManager("scenes")
    start = START_SCENE if START_SCENE in scene_manager.scenes else list(scene_manager.scenes)[0]
    scene_manager.set_scene(start)

    # input_device='xone' matches the Xone:24C; average L/R of the master mix.
    controller = Oculizer(PROFILE, scene_manager, input_device="xone",
                          average_dual_channels=True)
    controller.start()

    keymap = {}
    for name in OUR_SCENES:
        scene = scene_manager.scenes.get(name)
        if scene:
            keymap[str(scene.get("key_command", name))] = name

    print("\n=== Oculizer over Art-Net (profile: %s) ===" % PROFILE)
    for key, name in keymap.items():
        desc = scene_manager.scenes[name].get("description", "")
        print(f"  [{key}] {name:<14} {desc}")
    print("  [r] reload scenes from disk    [q] quit")
    print(f"\nActive scene: {start}\n")

    try:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "q":
                break
            elif cmd == "r":
                scene_manager.reload_scenes()
                print("scenes reloaded")
            elif cmd in keymap:
                controller.change_scene(keymap[cmd])
                print(f"Active scene: {keymap[cmd]}")
            elif cmd in scene_manager.scenes:
                controller.change_scene(cmd)
                print(f"Active scene: {cmd}")
            elif cmd:
                print(f"unknown command: {cmd!r}")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nstopping; sending blackout")
        controller.stop()
        controller.join()


if __name__ == "__main__":
    main()
