#!/usr/bin/env python3
"""Edit the presentation video: title cards, room tour, task clips with a wrist-camera inset, captions.

Inputs are the showcase recordings (tools/record_room_tour.py and `--showcase` runs of run_pick.py,
run_pick_place.py, run_handover.py). Outputs media/showcase.mp4 (1280x720, H.264) and a compact
media/showcase.gif for inline display on GitHub. Text is drawn with PIL (this ffmpeg has no drawtext).
Usage: python tools/make_showcase_video.py   (any python with numpy, pillow, imageio[ffmpeg])
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EP = ROOT / "artifacts" / "episodes"
W, H, FPS = 1280, 720, 30
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
BG = (18, 20, 24)
ACCENT = (118, 185, 0)  # NVIDIA-ish green, matches the README badges


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def card(title, subtitle="", seconds=2.2):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, H // 2 - 70, 14, H // 2 + 70], fill=ACCENT)
    size = 58  # shrink long titles (e.g. the repo URL) to fit the frame
    while size > 24 and d.textlength(title, font=font(size, True)) > W - 140:
        size -= 2
    d.text((80, H // 2 - 70), title, font=font(size, True), fill=(240, 240, 240))
    if subtitle:
        sub_size = 30
        while sub_size > 16 and d.textlength(subtitle, font=font(sub_size)) > W - 140:
            sub_size -= 2
        d.text((82, H // 2 + 12), subtitle, font=font(sub_size), fill=(170, 176, 186))
    frame = np.asarray(img)
    return [frame] * int(seconds * FPS)


def caption(frame, text, sub=""):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, H - 92, W, H], fill=(10, 12, 16, 170))
    d.rectangle([0, H - 92, 8, H], fill=ACCENT + (255,))
    d.text((30, H - 82), text, font=font(34, True), fill=(245, 245, 245))
    if sub:
        d.text((32, H - 40), sub, font=font(22), fill=(190, 196, 206))
    return np.asarray(img)


def inset(frame, small, label):
    """Picture-in-picture wrist view in the top-right corner."""
    img = Image.fromarray(frame)
    sw, sh = 352, 264
    pip = Image.fromarray(small).resize((sw, sh))
    x, y = W - sw - 24, 24
    d = ImageDraw.Draw(img)
    d.rectangle([x - 3, y - 3, x + sw + 2, y + sh + 2], outline=ACCENT, width=3)
    img.paste(pip, (x, y))
    d.text((x + 8, y + sh - 28), label, font=font(18, True), fill=(255, 255, 255))
    return np.asarray(img)


def read(path):
    return [f for f in imageio.mimread(path, memtest=False)]


def task_clip(run, ep, title, sub, wrist="cam_right_wrist", speed=1.0):
    show = read(EP / run / f"episode_{ep:04d}_showcase.mp4")
    pip = read(EP / run / f"episode_{ep:04d}_{wrist}.mp4")
    n = min(len(show), len(pip))
    idx = np.arange(0, n, speed).astype(int)
    label = "right wrist camera" if "right" in wrist else "left wrist camera"
    return [caption(inset(show[i], pip[i], label), title, sub) for i in idx]


def fade(frames, n=12):
    out = [f.astype(np.float32) for f in frames]
    for i in range(min(n, len(out) // 2)):
        a = (i + 1) / (n + 1)
        out[i] = out[i] * a + np.array(BG, np.float32) * (1 - a)
        out[-1 - i] = out[-1 - i] * a + np.array(BG, np.float32) * (1 - a)
    return [f.astype(np.uint8) for f in out]


def main():
    tour = read(ROOT / "artifacts" / "showcase" / "room_tour.mp4")
    seq = []
    seq += fade(card("Bimanual OpenArm in Isaac Sim", "picks · pick-and-place · handover · teleop · data recording", 3.0))
    seq += fade([caption(f, "A configurable room around the task table", "room preset 'full' · 21 drop-tested inventory objects as clutter") for f in tour])
    seq += fade(card("Position-based pick", "planned exactly offline · force-limited claw gripper"))
    seq += fade(task_clip("showcase_pick_mug", 0, "Pick: mug among clutter", "grasp across the body, away from the handle · 9/10 verified"))
    seq += fade(card("Pick-and-place", "into a KLT bin · random clutter each episode"))
    seq += fade(task_clip("showcase_place", 0, "Pick-and-place: soup can → bin", "10/10 verified · 3/3 with 6 clutter items"))
    seq += fade(card("Right → left handover", "right holds the top · left wraps the lower body"))
    seq += fade(task_clip("showcase_handover_mustard", 0, "Handover: mustard bottle", "right releases, left keeps the grip · 5/5 verified", wrist="cam_left_wrist"))
    seq += fade(card("Pick → handover → place", "right picks · hands over · left carries and drops it in the bin"))
    seq += fade(task_clip("showcase_transfer_mug", 0, "Pick → handover → bin: mug", "handle turned away from the left hand · 4 objects verified (mug 10/10)", wrist="cam_left_wrist"))
    seq += fade(card("github.com/amitgit24/Issac-sim-teleop-eval", "actively developed · LeRobot export · keyboard / gamepad teleop", 3.0))
    media = ROOT / "media"
    imageio.mimwrite(media / "showcase.mp4", seq, fps=FPS, codec="libx264", quality=None,
                     output_params=["-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                     macro_block_size=1)
    print(f"SHOWCASE VIDEO: {media / 'showcase.mp4'} ({len(seq) / FPS:.1f} s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
