#!/usr/bin/env python3
"""Validate the project model using PyCoral and one image."""
import argparse, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agrivision.config import load_config
from agrivision.inference import EdgeTpuInferencer

p=argparse.ArgumentParser()
p.add_argument('image',type=Path)
p.add_argument('--config',default=str(ROOT/'config.yaml'))
a=p.parse_args()
inf=EdgeTpuInferencer(load_config(a.config))
for pred in inf.predict(a.image,top_k=5):
    print(f"{pred.label}: {pred.confidence*100:.2f}%")
