from collections import Counter
import oblig1b_utils
import numpy as np
from language_identifier import LanguageIdentifier

train_data, test_data = oblig1b_utils.extract_wordlist()

print(type(test_data.IPA.values))

