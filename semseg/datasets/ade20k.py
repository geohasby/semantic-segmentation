import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class ADE20K(Dataset):
    CLASSES = ['human', 'building', 'animal',
               'vegetation', 'artefact', 'decoration']

    PALETTE = torch.tensor([[239, 67, 68], [113, 73, 157], [72, 188, 236], [
                           191, 218, 101], [250, 231, 137], [17, 79, 72]])

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        split = 'training' if split == 'train' else 'validation'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1

        img_path = Path(root) / 'images' / split
        self.files = list(img_path.glob('*.jpg'))

        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace(
            'images', 'annotations').replace('.jpg', '.png')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long() - 1


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    # visualize_dataset_sample(ADE20K, '/home/sithu/datasets/ADEChallenge/ADEChallengeData2016')
