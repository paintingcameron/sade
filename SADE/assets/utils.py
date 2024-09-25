import argparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import ast
import numpy as np
import torch
from random import shuffle, randint


def create_parts_grid(image_tensor, num_rows=None, num_columns=None, part_labels=None):
    num_parts = len(part_labels)

    images = []
    labels = []
    for k in range(len(image_tensor)):
        images.extend([tensor_to_pil(image_tensor[k,i,...], mode='L') for i in range(num_parts)])
        labels += [part_labels[i][k] for i in range(num_parts)]

    return create_image_grid(
        images,
        num_rows=num_rows,
        num_columns=num_columns,
        labels=labels
    )

def create_image_grid(image_list, num_rows=None, num_columns=None, labels=None, padding=5):
    num_images = len(image_list)

    if num_rows is None or num_columns is None:
        num_rows = int(num_images ** 0.5)  
        num_columns = (num_images + num_rows - 1) // num_rows

    image_width = image_list[0].width
    image_height = image_list[0].height
    label_height = 10 if labels else 0

    grid_width = num_columns * (image_width + padding) - padding
    grid_height = num_rows * (image_height + padding + label_height) - padding

    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    for i, image in enumerate(image_list):
        x = (i % num_columns) * (image_width + padding)
        y = (i // num_columns) * (image_height + padding + label_height) + label_height
        grid_image.paste(image, (x, y))

    if labels:
        draw = ImageDraw.Draw(grid_image)
        font = ImageFont.load_default()

        for i, label in enumerate(labels):
            x = (i % num_columns) * (image_width + padding) + (image_width - draw.textsize(label, font=font)[0]) // 2
            y = (i // num_columns) * (image_height + padding + label_height) - 1
            draw.text((x, y), label.replace(" ", "_"), fill=(0, 0, 0), font=font)

    return grid_image


class TupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, ast.literal_eval(values))
        except:
            parser.error(f"Invalid tuple argument: {values}")

class BooleanAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        try:
            v = None
            if isinstance(value, bool):
                v = value
            if value.lower() in ("yes", "true", "t", "y", "1"):
                v = True
            elif value.lower() in ("no", "false", "f", "n", "0"):
                v = False
            else:
                raise argparse.ArgumentTypeError("boolean value expected")

            setattr(namespace, self.dest, v)
        except:
            parser.error(f"Invalid boolean argument {value}")

def tensor_to_pil(tensor, mode="RGB"):
    '''mode = 'L' for greyscale'''
    arr = ((tensor.permute(1,2,0) + 1.0) * 127.5).cpu().detach().numpy()
    arr = np.clip(arr, 0, 255).astype("uint8")
    arr = arr.squeeze()
    return Image.fromarray(arr, mode=mode)

def clip_spaces(image, label):
    _, w, _ = image.shape
    char_width = get_char_width(w, len(label))

    while label[0] == ' ':
        label = label[1:]
        image = image[:,char_width:,:]
    while label[-1] == ' ':
        label = label[:-1]
        image = image[:,:-char_width,:]
    
    return image, label


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    snr = (alpha / sigma) ** 2
    return snr

def write_batch_to_cache(batch_labels, batch_images, cache, cnt, image_mode='L'):
    for img, label in zip(batch_images, batch_labels):
        imageKey = "image-%09d".encode() % (cnt+1)
        labelKey = "label-%09d".encode() % (cnt+1)

        img, label = clip_spaces(img, label)
        img = img.squeeze()
        
        with BytesIO() as f:
            Image.fromarray(img, mode=image_mode).save(f, format='png')
            image_data = f.getvalue()

        cache[imageKey] = image_data
        cache[labelKey] = label.encode()

        cnt += 1

    return cnt


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

    env.sync()


def get_char_width(image_width, num_chars):

    if image_width % num_chars != 0:
        l = int(np.floor((image_width % num_chars)/2.0))
        r = int(np.ceil((image_width % num_chars)/2.0))
        image_width = image_width - l - r

    return image_width // num_chars


def pad_image(img: Image, label, target_length):
    img = np.asarray(img)

    h, w, c = img.shape

    char_length = get_char_width(w, len(label))

    while w < char_length * target_length:
        left_padding = randint(0,1)

        padding_color = img[h//2, left_padding * (w-1), :][np.newaxis,np.newaxis, :]
        padded_image = np.ones((h,w + char_length,c), dtype=np.uint8) * padding_color
        padded_image[:, (char_length * left_padding) : (w + (char_length * left_padding)), :] = img
        img = padded_image

        label = " " + label if left_padding else label + " "

        _, w, _ = img.shape

    return Image.fromarray(img), label


part_length_lookup = {
    4:{1:4,2:3,3:2},
    5:{2:3,3:3},
    6:{2:4,3:4},
    7:{2:4,3:3,6:2,7:1},
}

overlap_lookup = {
    4:{1:0,2:2,3:1},
    5:{2:1,3:2},
    6:{2:2,3:3},
    7:{2:1,3:1,6:1,7:0}
}


def split_images(imgs, input_ids, num_parts, flatten_batch_dim=True, shuffle_output=True, device='cpu'):
    bsz,_,_,w = imgs.shape

    imgs = imgs.to(device)
    input_ids = input_ids.to(device)

    if num_parts == 1:
        input_ids = input_ids if flatten_batch_dim else input_ids.unsqueeze(0)
        return imgs, input_ids, imgs, torch.zeros((bsz,), dtype=torch.int32).to(device)

    num_chars = input_ids.shape[1]

    if w % num_chars != 0:
        l = int(np.floor((w % num_chars)/2.0))
        r = int(np.ceil((w % num_chars)/2.0))
        imgs = imgs[:,:,:,l:-r]

    _,_,_,w = imgs.shape
    
    char_parts = torch.chunk(imgs, num_chars, dim=3)
    char_parts = torch.stack(char_parts)
    
    prev = 0
    overlap = overlap_lookup[num_chars][num_parts]
    part_length = part_length_lookup[num_chars][num_parts]
    seg_indexes = list(range(num_chars))
    final_img_segs = []
    id_segs = []
    indexes = []
    for i in range(num_parts):
        seg_i = seg_indexes[prev:(prev+part_length)]
        prev = prev + part_length - overlap
        final_img_segs.append(torch.cat([char_parts[i] for i in seg_i], dim=3))
        id_segs.append(torch.cat([input_ids[:,i].reshape(1,-1) for i in seg_i], dim=0).transpose(0,1))
        indexes.append(torch.ones((bsz), dtype=torch.int32) * i)

    prev_segments = [torch.zeros_like(final_img_segs[0])]
    for parts in final_img_segs[:-1]:
        prev_segments.append(parts)

    if flatten_batch_dim:
        final_img_segs = torch.cat(final_img_segs, dim=0).to(device)
        prev_segments = torch.cat(prev_segments, dim=0).to(device)
        id_segs = torch.cat(id_segs, dim=0).to(device)
        indexes = torch.cat(indexes, dim=0).to(device)
    else:
        final_img_segs = torch.stack(final_img_segs, dim=0).to(device)
        prev_segments = torch.stack(prev_segments, dim=0).to(device)
        id_segs = torch.stack(id_segs, dim=0).to(device)

    if shuffle_output:
        shuffled_indecies = list(range(len(final_img_segs)))
        shuffle(shuffled_indecies)
        final_img_segs = final_img_segs[shuffled_indecies]
        prev_segments = prev_segments[shuffled_indecies]
        id_segs = id_segs[shuffled_indecies]
        indexes = indexes[shuffled_indecies]

    return final_img_segs, id_segs, prev_segments, indexes