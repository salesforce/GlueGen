import os
import torch
from torchvision import transforms
from data_coco_prepare import Imagelists_COCO
from data_laion_prepare import Imagelists_LAION
from torch.utils.data import Dataset
import json

import pdb

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))
    
def return_dataset(img_path, root, config):
#     img_path = 'coco-tr-pseudo-ldm-ddim50-s5-eta0'
#     root = 'outputs'
    img_TF = transforms.Compose([ResizeImage(256), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    
    
    dataset = Imagelists_COCO(img_path, root=root, transform=img_TF)
    
    
    print("%d sample in this dataset" % len(dataset))
    
    bs = config.data.params.batch_size
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs,
                                                num_workers=config.data.params.num_workers, shuffle=True,
                                                drop_last=True)
    return data_loader

# def return_dataset_laion(img_path, root, config):
# #     img_path = 'coco-tr-pseudo-ldm-ddim50-s5-eta0'
# #     root = 'outputs'
# #     img_TF = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# #     img_TF = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
#     img_TF = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.9, 1.0)), transforms.ToTensor()])
    
#     dataset = Imagelists_LAION(img_path, root=root, transform=img_TF)
    
#     print("%d sample in this dataset" % len(dataset))
    
#     bs = config.data.params.batch_size
    
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs,
#                                                 num_workers=config.data.params.num_workers, shuffle=True,
#                                                 drop_last=False, pin_memory=True)
    
# #     data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs,
# #                                                 num_workers=config.data.params.num_workers, shuffle=True,
# #                                                 drop_last=False, pin_memory=False, timeout=100)
#     return data_loader

import webdataset as wds
from torch.utils.data.dataloader import default_collate

class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)
            
class LaionDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "caption": self.text_processor(sample[1]["caption"]),
        }
    
def return_dataset_laion(img_path, root, config):
    normalize = transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )
    
    img_TF = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.9, 1.0)), transforms.ToTensor(),  normalize])
    
    dataset = LaionDataset(vis_processor=img_TF, text_processor=lambda x: x, location="/export/laion2b-data/laion2B-multi/data/part-00000/{00000..01743}.tar")
    
    # "/export/laion2b-data/laion2B-multi/data/part-00000/{00000..01743}.tar"
    
    print("%d sample in this dataset" % len(dataset))
    
    bs = config.data.params.batch_size
    
    data_loader = torch.utils.data.DataLoader(dataset.inner_dataset, batch_size=bs,
                                                num_workers=config.data.params.num_workers, shuffle=False,
                                                drop_last=False, pin_memory=True)
    
#     from torchvision import transforms

#     def to_image_text_pair(sample):
#         return sample[0], sample[1]["caption"]

    
    return data_loader


class capfilt_dataset(Dataset):
    def __init__(self, ann_path, transform):
#         self.ann = []
#         for file in ann_file:
#             print("loading %s" % file)
#             self.ann += json.load(open(file, "r"))
        f = open(ann_path,'r')
        self.ann = json.load(f)
        print("loading %s" % len(self.ann))
        self.transform = transform

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = ann["image"]
        image = Image.open(image_path).convert("RGB")
        
        data = {}
        
        if self.transform is not None:
            img = self.transform(img)
        
        data['image'] = (img - 0.5) * 2 #img #(img / 127.5 - 1.0).astype(np.float32)
        data['caption'] = ann["caption"]
        return data
    
def return_dataset_laion_all(img_path, config):
#     transform = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.9, 1.0)), transforms.ToTensor()])
    dataset = capfilt_dataset(img_path, transform)
    
    print("%d sample in this dataset" % len(dataset))
    
    bs = config.data.params.batch_size
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs,
                                                num_workers=config.data.params.num_workers, shuffle=True,
                                                drop_last=False, pin_memory=False)
    
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs,
#                                                 num_workers=config.data.params.num_workers, shuffle=True,
#                                                 drop_last=False, pin_memory=False, timeout=100)
    return data_loader