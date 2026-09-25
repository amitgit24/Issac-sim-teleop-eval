"""Teleop input: keyboard and gamepad (carb.input) plus a scripted source for headless tests.

Every source produces a TeleopCommand each control step: continuous twist (linear xyz, angular
roll/pitch/yaw, each in [-1, 1], world frame as seen from the robot: +x forward across the table,
+y to the robot's left, +z up) and one-shot events (gripper toggle, arm switch, record, ...).

Keyboard (hold to move, tap for events)          Gamepad (Xbox layout)
  W/S  forward/back     A/D  left/right            left stick      forward/back, left/right
  Q/E  up/down                                     right stick     up/down (vertical), yaw (horizontal)
  J/L  yaw left/right   I/K  pitch                 D-pad           pitch (up/down), roll (left/right)
  U/O  roll                                        A gripper  B switch arm  X precision  Y home
  SPACE gripper toggle  TAB switch arm             Start  record start / save   Back  discard
  P precision (30 %)    H home (ready pose)        RB     new episode (reset object)
  R record start  F finish+save  X discard  N new episode
"""

from dataclasses import dataclass, field

import numpy as np

EVENTS = ("gripper", "switch_arm", "precision", "home", "record", "save", "discard", "new_episode", "quit")


@dataclass
class TeleopCommand:
    linear: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [-1, 1] per axis
    angular: np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll, pitch, yaw in [-1, 1]
    events: set = field(default_factory=set)

    def merge(self, other: "TeleopCommand") -> "TeleopCommand":
        return TeleopCommand(
            np.clip(self.linear + other.linear, -1, 1), np.clip(self.angular + other.angular, -1, 1), self.events | other.events
        )


class _Source:
    def poll(self) -> TeleopCommand:
        raise NotImplementedError

    def close(self):
        pass


class KeyboardSource(_Source):
    """Held keys give motion, taps give events. Needs the Kit app window (GUI, or headless for tests)."""

    AXES = {  # key -> (vector, index, sign)
        "W": ("linear", 0, +1), "S": ("linear", 0, -1),
        "A": ("linear", 1, +1), "D": ("linear", 1, -1),
        "Q": ("linear", 2, +1), "E": ("linear", 2, -1),
        "U": ("angular", 0, +1), "O": ("angular", 0, -1),
        "I": ("angular", 1, +1), "K": ("angular", 1, -1),
        "J": ("angular", 2, +1), "L": ("angular", 2, -1),
    }
    TAPS = {
        "SPACE": "gripper", "TAB": "switch_arm", "P": "precision", "H": "home",
        "R": "record", "F": "save", "X": "discard", "N": "new_episode", "ESCAPE": "quit",
    }

    def __init__(self):
        import carb
        import omni.appwindow

        self._carb = carb
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._held: set[str] = set()
        self._events: set[str] = set()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event, *args, **kwargs):
        et = self._carb.input.KeyboardEventType
        name = event.input.name
        if event.type == et.KEY_PRESS:
            if name in self.AXES:
                self._held.add(name)
            elif name in self.TAPS:
                self._events.add(self.TAPS[name])
        elif event.type == et.KEY_RELEASE:
            self._held.discard(name)
        return True

    def poll(self) -> TeleopCommand:
        cmd = TeleopCommand()
        for name in self._held:
            vec, i, sign = self.AXES[name]
            getattr(cmd, vec)[i] += sign
        cmd.events, self._events = self._events, set()
        cmd.linear, cmd.angular = np.clip(cmd.linear, -1, 1), np.clip(cmd.angular, -1, 1)
        return cmd

    def close(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None


class GamepadSource(_Source):
    """First connected gamepad (Xbox layout via carb.input). Sticks are analog with a dead zone."""

    DEAD_ZONE = 0.12

    def __init__(self, index: int = 0, pad=None):
        """index: which connected gamepad; pad: an explicit carb gamepad handle (e.g. a virtual one in tests)."""
        import carb
        import omni.appwindow

        self._carb = carb
        gi = carb.input.GamepadInput
        carb.settings.get_settings().set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        self._input = carb.input.acquire_input_interface()
        self._pad = pad if pad is not None else omni.appwindow.get_default_app_window().get_gamepad(index)
        # analog: input -> (vector, index, sign)
        self.AXES = {
            gi.LEFT_STICK_UP: ("linear", 0, +1), gi.LEFT_STICK_DOWN: ("linear", 0, -1),
            gi.LEFT_STICK_LEFT: ("linear", 1, +1), gi.LEFT_STICK_RIGHT: ("linear", 1, -1),
            gi.RIGHT_STICK_UP: ("linear", 2, +1), gi.RIGHT_STICK_DOWN: ("linear", 2, -1),
            gi.RIGHT_STICK_LEFT: ("angular", 2, +1), gi.RIGHT_STICK_RIGHT: ("angular", 2, -1),
            gi.DPAD_UP: ("angular", 1, +1), gi.DPAD_DOWN: ("angular", 1, -1),
            gi.DPAD_LEFT: ("angular", 0, +1), gi.DPAD_RIGHT: ("angular", 0, -1),
        }
        self.BUTTONS = {
            gi.A: "gripper", gi.B: "switch_arm", gi.X: "precision", gi.Y: "home",
            gi.MENU2: "record_or_save", gi.MENU1: "discard", gi.RIGHT_SHOULDER: "new_episode",
        }
        self._values = {k: 0.0 for k in self.AXES}
        self._pressed: set = set()
        self._events: set[str] = set()
        self.recording = False  # Start toggles record -> save, tracked by the app via events
        self._sub = self._input.subscribe_to_gamepad_events(self._pad, self._on_event)

    def _on_event(self, event, *args, **kwargs):
        v = float(event.value)
        if event.input in self._values:
            self._values[event.input] = 0.0 if abs(v) < self.DEAD_ZONE else v
        elif event.input in self.BUTTONS:
            if v > 0.5 and event.input not in self._pressed:  # rising edge
                self._pressed.add(event.input)
                self._events.add(self.BUTTONS[event.input])
            elif v <= 0.5:
                self._pressed.discard(event.input)
        return True

    def poll(self) -> TeleopCommand:
        cmd = TeleopCommand()
        for inp, v in self._values.items():
            vec, i, sign = self.AXES[inp]
            getattr(cmd, vec)[i] += sign * v
        events, self._events = self._events, set()
        if "record_or_save" in events:  # one button: start recording, then save
            events.discard("record_or_save")
            events.add("save" if self.recording else "record")
        cmd.events = events
        cmd.linear, cmd.angular = np.clip(cmd.linear, -1, 1), np.clip(cmd.angular, -1, 1)
        return cmd

    def close(self):
        if self._sub is not None:
            self._input.unsubscribe_to_gamepad_events(self._pad, self._sub)
            self._sub = None


class ScriptedSource(_Source):
    """Replays a list of (duration_s, TeleopCommand-like dict) for headless tests and demos."""

    def __init__(self, script, dt: float):
        self._steps = []
        for duration, spec in script:
            n = max(1, int(round(duration / dt)))
            base = TeleopCommand(np.array(spec.get("linear", [0, 0, 0]), float), np.array(spec.get("angular", [0, 0, 0]), float))
            ev = set(spec.get("events", []))
            self._steps.append(TeleopCommand(base.linear, base.angular, ev))
            self._steps.extend(TeleopCommand(base.linear, base.angular, set()) for _ in range(n - 1))
        self._i = 0

    @property
    def done(self) -> bool:
        return self._i >= len(self._steps)

    def poll(self) -> TeleopCommand:
        if self.done:
            return TeleopCommand(events={"quit"})
        cmd = self._steps[self._i]
        self._i += 1
        return TeleopCommand(cmd.linear.copy(), cmd.angular.copy(), set(cmd.events))


class MultiSource(_Source):
    def __init__(self, sources):
        self.sources = list(sources)

    def poll(self) -> TeleopCommand:
        cmd = TeleopCommand()
        for s in self.sources:
            cmd = cmd.merge(s.poll())
        return cmd

    def close(self):
        for s in self.sources:
            s.close()
