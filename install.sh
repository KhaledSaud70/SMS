#!/usr/bin/bash
conda create -n sms python=3.10
conda activate sms
conda install pytorch torchvision scikit-learn -c pytorch
conda install pytz python-dateutil
pip install open_clip_torch