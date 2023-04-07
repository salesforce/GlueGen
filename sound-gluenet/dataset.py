from torch.utils.data import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, text):
        self.text = text
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        text = self.text[idx]
        sample = {"Text": text}
        return sample