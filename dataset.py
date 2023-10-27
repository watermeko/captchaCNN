from torch.utils.data import Dataset
from PIL import Image
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import json

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform, characters):
        self.file_list = []
        files = os.listdir(data_dir)
        for file in files:
            path = os.path.join(data_dir, file)
            self.file_list.append(path)
        self.transform = transform

        self.char2int = {}
        for i, char in enumerate(characters):
            self.char2int[char] = i
            
    def char_to_int(self,char):
        return self.char2int[char]

    def __len__(self):
        return len(self.file_list)

    # get the data and label
    def __getitem__(self, index):
        if index > len(self):
            return
        path = self.file_list[index]
        img = Image.open(path).convert('L')
        # Convet image to tensor data
        img = self.transform(img)

        label_char = os.path.basename(path).split('_')[0]
        label = []
        for char in label_char:
            label.append(self.char_to_int(char))
        label = torch.tensor(label)
        return img, label


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    height = config["resize_height"]
    width = config["resize_width"]
    train_data_path = config["train_data_path"]
    characters = config["characters"]
    batch_size = config["batch_size"]
    epochbs = config["epoch_num"]

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    
    dataset = CaptchaDataset(train_data_path, transform, characters)
    data_load = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochbs):
        print("epoch = %d"%(epoch))
        for batch_idx, (data, label) in enumerate(data_load):
            print("batch_idx = %d label = %s"%(batch_idx, label))
    

