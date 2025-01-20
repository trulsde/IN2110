import json

class Data:
    def __init__(self, documents, split):
        self._documents = documents
        self._split = split
        self._categories = ['games', 'restaurants', 'literature']
        self._data = []
        self._labels = []
        for ordbok in documents:
            if ordbok['metadata']['category'] in self._categories and ordbok['metadata']['split'] == self._split:
                self._data.append(ordbok['text'])
                self._labels.append(ordbok['metadata']['category'])

        assert len(self._data) == len(self._labels)

    def __str__(self):
        return self._split

    def tilfeller_av_hver_kategori(self):
        oversikt = {}
        for kategori1 in self._categories:
            matcher = 0
            for kategori2 in self._labels:
                if kategori1 == kategori2:
                    matcher += 1
            oversikt[kategori1] = matcher

        return oversikt






