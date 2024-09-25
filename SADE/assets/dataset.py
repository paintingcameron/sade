import os
import os.path
import lmdb
import six

from PIL import Image

from torch.utils.data import Dataset
from .utils import pad_image

class LMDBImageDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, label_transform=None, label_length=0):
        
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.transform = transform
        self.label_transform = label_transform
        self.label_length = label_length

        with self.env.begin() as txn:
            self.length = int(txn.get('num-samples'.encode()))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        index += 1 #lmdb starts at 1

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index

            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')  

                if len(label) < self.label_length:
                    img, label = pad_image(img, label, self.label_length)

                if self.transform is not None:
                    img = self.transform(img)

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                label = '-'

            if self.label_transform is not None:
                label = self.label_transform(label)

        return (img, label)