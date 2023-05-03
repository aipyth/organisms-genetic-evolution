import random

consonants = [
    'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't',
    'v', 'w', 'x', 'y', 'z'
]
vowels = ['a', 'e', 'i', 'o', 'u']


def generate_name(num_syllables):
    name = ''
    for i in range(num_syllables):
        if i % 2 == 0:  # Choose a consonant for even syllables
            name += random.choice(consonants)
        else:  # Choose a vowel for odd syllables
            name += random.choice(vowels)
    return name.capitalize()  # Capitalize the first letter