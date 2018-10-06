# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:19:12 2018

@author: haoyu
"""

import os.path
os.cpu_count()

#第三方库安装脚本
import os
libs={'numpy','matplotlib','pillow','request'}
try:
    for lib in libs:
        os.system('pip install'+lib)
    print('Successful')
except:
    print('Failed Somehow')