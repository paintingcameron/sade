
import sys
import os
import json
from random import choice
import concurrent.futures
import argparse


sys.path.append("./")
from datagen.data_creation.fonts import Fonts, get_font_str
from datagen.simpleImageSets import (
    VenezualaNumberPlate,
)
from datagen.data_creation.wordgen import (
    get_positioned_tuple_words,
    get_permutated_words,
    missing_char_position,
    random_with_replacement,
    random_words_missing_char,
    random_words_present_char,
)

parser = argparse.ArgumentParser()

parser.add_argument("dataset_path")

args = parser.parse_args()

dataset_path = args.dataset_path

image_sets = [VenezualaNumberPlate]
fonts = [Fonts.FONTS['UBUNTU_MONO']]
text_colors = [
    (0, 0, 0, 255),
]
background_colors = [
    (50, 50, 50, 255),
    (171, 171, 171),
]
padding = 10

alphabet = "ALMQX23456"
word_length = 5

class_cond = False

repeats = 1
train_size = 3000
val_size = 200
test_size = 200

POSITIONAL_TUPLES = 0
RANDOM = 1
PERUMUTATIONS = 2
MISSING_CHAR = 3
RANDOM_WITH_REPLACE = 4
RANDOM_MISSING_CHAR = 5
PRESENT_CHAR = 6

train_gen_type = RANDOM_WITH_REPLACE
# val_gen_type = RANDOM
# test_gen_type = RANDOM

tuple_size = 2
missing_char = 'L'
missing_pos = 2
blank_images = 0

def get_words(gen_type, num_words=None):
    words = []
    if gen_type == POSITIONAL_TUPLES:
        words = get_positioned_tuple_words(alphabet=alphabet, word_length=word_length, tuple_size=tuple_size)
        num_words = len(words)
    elif gen_type == RANDOM:
        words = get_permutated_words(alphabet=alphabet, word_length=word_length, shuffled=True)
    elif gen_type == PERUMUTATIONS:
        words = get_permutated_words(alphabet=alphabet, word_length=word_length)
    elif gen_type == MISSING_CHAR:
        words = missing_char_position(alphabet=alphabet, word_length=word_length, char=missing_char, position=missing_pos)
    elif gen_type == RANDOM_WITH_REPLACE:
        words = random_with_replacement(alphabet=alphabet, word_length=word_length, num_words=num_words)
    elif gen_type == RANDOM_MISSING_CHAR:
        words = random_words_missing_char(alphabet=alphabet, word_length=word_length, char=missing_char, position=missing_pos, num_words=num_words)
    elif gen_type == PRESENT_CHAR:
        words = random_words_present_char(alphabet=alphabet, word_length=word_length, char=missing_char, position=missing_pos, num_words=num_words)

    if num_words is not None and num_words <= len(words):
        words = words[:num_words]

    if blank_images > 0:
        words += [""] * blank_images

    return words

def create_image(text):
    image_set = choice(image_sets)
    font = Fonts.getFont(choice(fonts))
    text_color = choice(text_colors)
    background_color = choice(background_colors)

    args = {}
    args['background_color'] = background_color

    image = image_set.create(
        text=text,
        font=font,
        **args
    )

    return image

def create_and_save(text, stage, i):
    image = create_image(text)

    if class_cond:
        file_dir = os.path.join(stage, text)
        file_name = f"image-{i}.png"
    else:
        file_dir = os.path.join(stage)
        file_name = f"{text}_image-{i}.png"

    if not os.path.exists(os.path.join(dataset_path, file_dir)):
        os.makedirs(os.path.join(dataset_path, file_dir), exist_ok=True)

    file_path = os.path.join(dataset_path, file_dir, file_name)
    image.save(file_path, format="PNG")
    return stage, os.path.join(file_dir, file_name), text

def gen_words():
    available_words = get_permutated_words(alphabet=alphabet, word_length=word_length, shuffled=True)

    train_words = get_words(train_gen_type, train_size)
    available_words = [w for w in available_words if w not in train_words]

    val_words = available_words[:val_size]

    if test_size is None:
        test_words = available_words[val_size:]
    elif test_size > 0:
        available_words = [w for w in available_words if w not in val_words]
        test_words = available_words[:test_size]
    else:
        test_words = []

    return train_words, val_words, test_words

def create_images():
    train_words, val_words, test_words = gen_words()

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executer:
        futures = []
        for r in range(repeats):
            for words, stage in zip((train_words, val_words, test_words), ("train", "val", "test")):
                for i, word in enumerate(words):
                    futures.append(executer.submit(create_and_save, word, stage, r*len(words) + i))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()

            print(f"Saved image: {result[1]}")
            results.append(result)

    return results

if __name__ == "__main__":
    stages = ['train', 'val', 'test']
    for stage in stages:
        os.makedirs(os.path.join(dataset_path, stage), exist_ok=True)

    results = create_images()

    lengths = []

    for stage, gtFile in zip(('train', 'val', 'test'), ('gtTraining.txt', 'gtValidation.txt', 'gtTest.txt')):
        details = [r[1:] for r in results if r[0] == stage]

        lengths.append(len(details))

        with open(os.path.join(dataset_path, gtFile), 'w') as f:
            for path, word in details:
                f.write(f"{path}\t{word}\n")

    args = {
        "alphabet": alphabet,
        "word_length": word_length,
        "class_cond": class_cond,
        "repeats": repeats,
        "train_size": lengths[0],
        'val_size': lengths[1],
        'test_size': lengths[2],
        "tuple_length": tuple_size,
        "train_gen_type": train_gen_type,
        "text_colors": text_colors,
        "backgroun_colors": background_colors,
        "padding": padding,
        "fonts": [get_font_str(font) for font in fonts],
        "image_sets": [image_set.__name__ for image_set in image_sets],
    }

    with open(os.path.join(dataset_path, "config.json"), 'w') as f:
        json.dump(args, f, indent=4)