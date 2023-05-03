'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
 * Modified from Stable Diffusion repo: https://github.com/CompVis/stable-diffusion
 * Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors
'''

import cv2
import fire
from imwatermark import WatermarkDecoder


def testit(img_path):
    bgr = cv2.imread(img_path)
    decoder = WatermarkDecoder('bytes', 136)
    watermark = decoder.decode(bgr, 'dwtDct')
    try:
        dec = watermark.decode('utf-8')
    except:
        dec = "null"
    print(dec)


if __name__ == "__main__":
    fire.Fire(testit)