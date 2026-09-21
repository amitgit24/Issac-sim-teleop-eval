#!/usr/bin/env python3
"""Contact sheet of one recorded episode: one column per phase (end of phase), one row per camera.

Usage: python tools/episode_contact_sheet.py <episode_dir> <episode_index> [out.png]
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

d, ep = Path(sys.argv[1]), int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else str(d / f"episode_{ep:04d}_sheet.png")
npz = np.load(d / f"episode_{ep:04d}.npz")
phase = npz["phase"].astype(int)
names = [str(n) for n in npz["phase_names"]] if "phase_names" in npz else ["approach", "descend", "close", "settle", "lift", "hold"]
idx = [0] + [int(np.where(phase == p)[0][-1]) for p in range(phase.max() + 1)]
labels = ["start"] + [f"end {names[p]}" for p in range(phase.max() + 1)]
cams = [c for c in ("cam_high", "cam_right_wrist", "cam_left_wrist") if (d / f"episode_{ep:04d}_{c}.mp4").exists()]
W, H = (240, 180) if len(names) > 8 else (320, 240)
sheet = Image.new("RGB", (W * len(idx), (H + 16) * len(cams)), "white")
dr = ImageDraw.Draw(sheet)
for r, c in enumerate(cams):
    frames = imageio.mimread(d / f"episode_{ep:04d}_{c}.mp4", memtest=False)
    for k, (i, lab) in enumerate(zip(idx, labels)):
        sheet.paste(Image.fromarray(frames[min(i, len(frames) - 1)]).resize((W, H)), (k * W, r * (H + 16) + 16))
        dr.text((k * W + 4, r * (H + 16) + 2), f"{c}: {lab} (t={i / 30:.1f}s)", fill="black")
sheet.save(out)
print(out)
