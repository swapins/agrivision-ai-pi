#!/usr/bin/env python3
"""Publish release artifacts to a Hugging Face model repository.

Authenticate outside the script with `hf auth login`; do not place tokens in
source code or command history.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
from huggingface_hub import HfApi, create_repo


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-id',required=True,help='e.g. username/agrivision-mobilenetv2-edge-tpu')
    p.add_argument('--artifact-dir',type=Path,default=Path('training/output/agrivision-mobilenetv2'))
    p.add_argument('--private',action='store_true')
    a=p.parse_args()
    release=a.artifact_dir/'hf_release'
    release.mkdir(parents=True,exist_ok=True)
    wanted=[
        'plant_health_int8.tflite','labels.txt','model_manifest.json','README.md'
    ]
    for name in wanted:
        src=a.artifact_dir/name
        if src.exists(): shutil.copy2(src,release/name)
    candidates=list((a.artifact_dir/'edgetpu').glob('*_edgetpu.tflite')) if (a.artifact_dir/'edgetpu').exists() else []
    if candidates: shutil.copy2(candidates[0],release/'plant_health_edgetpu.tflite')
    if not (release/'README.md').exists():
        template=Path(__file__).with_name('model-card-template.md')
        shutil.copy2(template,release/'README.md')
    create_repo(a.repo_id,repo_type='model',private=a.private,exist_ok=True)
    api=HfApi()
    api.upload_folder(repo_id=a.repo_id,repo_type='model',folder_path=release,commit_message='Publish AgriVision AI model release')
    print(f'Published https://huggingface.co/{a.repo_id}')

if __name__=='__main__': main()
