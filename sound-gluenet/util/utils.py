'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
 
 * Redistributed from repo: https://github.com/AndreyGuzhov/AudioCLIP
 * Copyright (c) 2021 Andrey Guzhov. MIT License
'''

from audioclip.ignite_trainer import _utils
import torchvision as tv
from typing import Type
import json

#----------------------------------------------------------------------------

def get_dataloader(config_path, params):
    config = json.load(open(config_path))
    
    transforms = config['Transforms'] 

    transforms_train = list()
    transforms_test = list()

    for idx, transform in enumerate(transforms):
        use_train = transform.get('train', True)
        use_test = transform.get('test', True)

        transform = _utils.load_class(transform['class'])(**transform['args'])

        if use_train:
            transforms_train.append(transform)
        if use_test:
            transforms_test.append(transform)

        transforms[idx]['train'] = use_train
        transforms[idx]['test'] = use_test

    transforms_train = tv.transforms.Compose(transforms_train)
    transforms_test = tv.transforms.Compose(transforms_test)

    dataset_class = config['Dataset']['class']
    dataset_args = config['Dataset']['args']
    Dataset: Type = _utils.load_class(dataset_class)
        
    batch_train = params["TRAIN_BATCH_SIZE"]
    batch_test = params["TRAIN_BATCH_SIZE"]
        
    workers_train = 2
    workers_test = 2

    train_loader, eval_loader = _utils.get_data_loaders(
        Dataset,
        dataset_args,
        batch_train,
        batch_test,
        workers_train,
        workers_test,
        transforms_train,
        transforms_test
    )

    return train_loader, eval_loader