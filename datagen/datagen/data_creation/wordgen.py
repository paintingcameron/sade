from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from random import shuffle, randint, sample, choice
from itertools import product
from multiprocessing import Pool


def get_positioned_tuple_words(alphabet, word_length, tuple_size):
    alphabet = list(set(alphabet))
    alphabet.sort()

    num_pos_tuple_sets = word_length - (tuple_size - 1)
    positioned_tuple_sets = [["".join(w) for w in product(alphabet, repeat=tuple_size)] 
                       for _ in range(num_pos_tuple_sets)]
    overlap_length = tuple_size - 1
    
    words = []
    while len(positioned_tuple_sets[0]) != 0:
        word = positioned_tuple_sets[0].pop(randint(0, len(positioned_tuple_sets[0])-1))
        for tuples_set in positioned_tuple_sets[1:]:
            available_tuples = [tp for tp in tuples_set if tp[:overlap_length] == word[-overlap_length:]]
            tp = available_tuples.pop(randint(0,len(available_tuples)-1))
            tuples_set.remove(tp)
            word = f"{word}{tp[-1]}"
        words.append(word)

    return words

def get_permutated_words(alphabet, word_length, shuffled=False):
    words = ["".join(i) for i in product(alphabet, repeat=word_length)]

    if shuffled:
        shuffle(words)

    return words

def missing_char_position(alphabet, word_length, char, position):
    words = ["".join(i) for i in product(alphabet, repeat=word_length)]
    shuffle(words)

    words = [word for word in words if word[position] != char]

    return words

def random_words_missing_char(alphabet, word_length, char, position, num_words):

    words = []
    while len(words) < num_words:
        word = "".join([choice(alphabet) for _ in range(word_length)])

        if word[position] != char:
            words.append(word)

    return words

def random_words_present_char(alphabet, word_length, char, position, num_words):
    words = []
    while len(words) < num_words:
        word = [choice(alphabet) for _ in range(word_length)]
        word[position] = char
        word = "".join(word)

        words.append(word)

    return words

def generate_random_word(alphabet, word_length):
    return "".join([choice(alphabet) for _ in range(word_length)])

def generate_random_words(alphabet, word_length, num_words):
    return [generate_random_word(alphabet, word_length) for _ in range(num_words)]

def random_with_replacement(alphabet, word_length, num_words, num_processes=4):
    chunk_size = num_words // num_processes
    chunks = [(alphabet, word_length, chunk_size) for _ in range(num_processes - 1)]
    chunks.append((alphabet, word_length, num_words - chunk_size * (num_processes - 1)))

    with Pool(num_processes) as pool:
        results = pool.starmap(generate_random_words, chunks)

    words = [word for sublist in results for word in sublist]

    return words