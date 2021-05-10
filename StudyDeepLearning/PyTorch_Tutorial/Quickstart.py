#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Quickstart.py: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import os
import sys
import platform

# import library for Quickstart
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


# StudyDeepLearning
# Date: 2021/05/09
# Filename: Quickstart 

def _information():
    print(f'current directory: {os.getcwd()}')
    print(f'python version: {sys.version}')
    print(f'OS version: {platform.platform()}')


def main():
    _information()
    return


if __name__ == '__main__':
    main()
