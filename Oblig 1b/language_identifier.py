import pandas as pd
import sklearn.linear_model
import numpy as np

class LanguageIdentifier:

    def __init__(self):
        self._model = self._model = sklearn.linear_model.LogisticRegression(solver="liblinear", multi_class='ovr')
        self._symbols = None

    def _extract_unique_symbols(self, transcriptions, min_nb_occurrences=10):

        min_occurrences = min_nb_occurrences
        occurrence_table = {}
        symbol_list = []

        assert isinstance(transcriptions, list), "Parameter 'transcriptions' must be a list"

        for transcription in transcriptions:
            for character in transcription:
                if character in occurrence_table:
                    occurrence_table[character] += 1
                else:
                    occurrence_table[character] = 1

        for key in occurrence_table:
            if occurrence_table[key] >= min_occurrences:
                symbol_list.append(key)

        self._symbols = symbol_list

    def _extract_feats(self, transcriptions):

        n = len(transcriptions)
        m = len(self._symbols)

        X = np.zeros((n, m))

        for i in range(len(transcriptions)):
            for j in range(len(self._symbols)):
                if self._symbols[j] in transcriptions[i]:
                    X[i][j] = 1
        return X

    def train(self, transcriptions, languages):

        assert isinstance(transcriptions, list), "Parameter 'transcriptions' must be a list"
        assert isinstance(languages, list), "Parameter 'languages' must be a list"
        assert len(transcriptions) == len(
            languages), "Transcriptions list and languages (labels) list must be of same length)"

        """
        Initierer ei liste med unike språknavn i parameterlista languages, ei liste
        med heltall mellom 0 og len(unique_languages) (antall unike språk). Hvert språk
        får da sitt unike tall som klasseetikett. For ordens skyld lager jeg også ei ordbok
        som har disse unike tall-klasseetikettene som nøkler og språknavn som tall. 
        """
        unique_languages = pd.Series(languages).unique().tolist()
        int_class_labels = []
        class_label_dict = {}

        """
        Den første for-løkka sørger for at lista int_class_labels blir lik som parameterlista
        'languages', bare at klassene (språkene) er byttet ut med tall. Tallet tilsvarer
        språkets indeks i lista unique_languages, som bare er lista over unike språk i 
        datasettet vårt.
        """
        for i in range(len(languages)):
            label = unique_languages.index(languages[i])
            int_class_labels.append(label)

        for i in range(len(unique_languages)):
            class_label_dict[i] = unique_languages[i]

        self._model.fit(self._extract_feats(transcriptions), int_class_labels)