#!/usr/bin/env python3
"""Inject real carb keyboard / gamepad events (headless) and check the teleop input layer's commands.

Covers the key and button map in scenes/teleop_input.py without a physical device.
Usage: /data/isaac/isaacsim/bin/python tests/check_teleop_inputs.py   (exit 1 on any mismatch)
"""
import sys
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import env_common as E  # noqa: E402

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402

from teleop_input import GamepadSource, KeyboardSource  # noqa: E402

KI, KT, GI = carb.input.KeyboardInput, carb.input.KeyboardEventType, carb.input.GamepadInput


def main():
    provider = carb.input.acquire_input_provider()
    win = omni.appwindow.get_default_app_window()
    kb = win.get_keyboard()
    failures = []

    def expect(label, got, want):
        ok = np.allclose(got, want) if isinstance(want, (list, np.ndarray)) else got == want
        print(f"{'OK  ' if ok else 'FAIL'} {label}: got {got}, want {want}", flush=True)
        if not ok:
            failures.append(label)

    def pump():
        for _ in range(3):
            simulation_app.update()

    # --- keyboard
    ks = KeyboardSource()
    for key, vec, idx, sign in (("W", "linear", 0, 1), ("D", "linear", 1, -1), ("Q", "linear", 2, 1), ("J", "angular", 2, 1), ("K", "angular", 1, -1)):
        provider.buffer_keyboard_key_event(kb, KT.KEY_PRESS, getattr(KI, key), 0)
        pump()
        want = np.zeros(3)
        want[idx] = sign
        expect(f"key {key} held", getattr(ks.poll(), vec), want)
        provider.buffer_keyboard_key_event(kb, KT.KEY_RELEASE, getattr(KI, key), 0)
        pump()
        expect(f"key {key} released", getattr(ks.poll(), vec), np.zeros(3))
    for key, event in (("SPACE", "gripper"), ("TAB", "switch_arm"), ("R", "record"), ("F", "save"), ("X", "discard"), ("N", "new_episode"), ("H", "home"), ("P", "precision")):
        provider.buffer_keyboard_key_event(kb, KT.KEY_PRESS, getattr(KI, key), 0)
        provider.buffer_keyboard_key_event(kb, KT.KEY_RELEASE, getattr(KI, key), 0)
        pump()
        expect(f"key {key} tap", ks.poll().events, {event})
    expect("events are one-shot", ks.poll().events, set())
    ks.close()

    # --- gamepad: no physical pad on a headless machine, so create a virtual one and connect it
    pad = provider.create_gamepad("teleop-test-pad", "00000000-0000-0000-0000-000000000001")
    provider.set_gamepad_connected(pad, True)
    gs = GamepadSource(pad=pad)
    provider.buffer_gamepad_event(pad, GI.LEFT_STICK_UP, 0.8)
    pump()
    expect("left stick up = forward", gs.poll().linear, [0.8, 0, 0])
    provider.buffer_gamepad_event(pad, GI.LEFT_STICK_UP, 0.05)  # inside the dead zone
    pump()
    expect("dead zone", gs.poll().linear, [0, 0, 0])
    provider.buffer_gamepad_event(pad, GI.RIGHT_STICK_RIGHT, 1.0)
    pump()
    expect("right stick right = yaw right", gs.poll().angular, [0, 0, -1])
    provider.buffer_gamepad_event(pad, GI.RIGHT_STICK_RIGHT, 0.0)
    for button, event in ((GI.A, "gripper"), (GI.B, "switch_arm"), (GI.Y, "home"), (GI.MENU1, "discard")):
        provider.buffer_gamepad_event(pad, button, 1.0)
        provider.buffer_gamepad_event(pad, button, 0.0)
        pump()
        expect(f"button {button.name}", gs.poll().events, {event})
    for rec, want in ((False, "record"), (True, "save")):
        gs.recording = rec
        provider.buffer_gamepad_event(pad, GI.MENU2, 1.0)
        provider.buffer_gamepad_event(pad, GI.MENU2, 0.0)
        pump()
        expect(f"Start while recording={rec}", gs.poll().events, {want})
    gs.close()
    print("TELEOP INPUT CHECK " + ("PASSED" if not failures else f"FAILED: {failures}"), flush=True)
    if failures:
        raise RuntimeError("teleop input mismatches")


if __name__ == "__main__":
    E.run_main(main, simulation_app)
