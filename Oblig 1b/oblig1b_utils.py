import urllib.request
import pandas
import sklearn.model_selection
import os, re, random

#####################################
# METODER FOR LOGISTISK REGRESJON
#####################################

ORDFILER = {"norsk":"https://github.com/open-dict-data/ipa-dict/blob/master/data/nb.txt?raw=true",
        "arabisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ar.txt?raw=true",
        "finsk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fi.txt?raw=true",
        "patwa":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/jam.txt?raw=true",
        "farsi":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fa.txt?raw=true",
        "tysk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/de.txt?raw=true",
        "engelsk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_UK.txt?raw=true",
        "rumensk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ro.txt?raw=true",
        "khmer":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/km.txt?raw=true",
        "fransk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fr_FR.txt?raw=true",
        "japansk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ja.txt?raw=true",
        "spansk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/es_ES.txt?raw=true",
         "svensk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/sv.txt?raw?true",
         "koreansk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ko.txt?raw?true",
         "swahilisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/sw.txt?raw?true",
         "vietnamesisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/vi_C.txt?raw?true",
        "mandarin":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/zh_hans.txt?raw?true",
        "malayisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ma.txt?raw?true",
        "kantonesisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/yue.txt?raw?true",
         "islandsk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/is.txt?raw=true"}


        
def extract_wordlist(cache_file="./langid_data.csv"):
    """Returnerer 2 Dataframes (en for trening og en for testing) hvor hver rekke 
    tilsvarer et ord med dets phonetisk transkripsjon og språket det hører til.
    Hvis data ikke er allerede lastet ned vil metoden laster data fra github."""
    
    
    # If the wordlists are already downloaded, we simply retrieve the cached file
    if cache_file is not None and os.path.exists(cache_file):
        print("Reading cached file from %s"%cache_file)
        full_wordlist = pandas.read_csv(cache_file)
    
    # Otherwise, we download the data from github       
    else:      
        full_wordlist = _download_wordlist()
        if cache_file is not None:
            full_wordlist.to_csv(cache_file)
            
    # Lage et treningssett og en testsett (med 10% av data)
    train, test = sklearn.model_selection.train_test_split(full_wordlist, test_size=0.1)            
        
    print("Treningsett: %i eksempler, testsett: %i eksempler"%(len(train), len(test)))
    return train, test


def _download_wordlist(max_nb_words_per_language=50000):
    """Laster ned fra Github ordlister med ord og deres phonetiske transkripsjoner 
    i flere språk."""
    
    full_wordlist = []
    for lang, wordfile in ORDFILER.items():
        
        print("Nedlasting av ordisten for", lang, end="... ")
        data = urllib.request.urlopen(wordfile)
        
        wordlist_for_language = []
        for linje in data:
            linje = linje.decode("utf8").rstrip("\n")
            word, transcription = linje.split("\t")
            
            # Noen transkripsjoner har feil tegn for "primary stress"
            transcription = transcription.replace("\'", "ˈ")
            
            # vi tar den første transkripsjon (hvis det finnes flere) 
            # og fjerner slashtegnene ved start og slutten
            match = re.match("/(.+?)/", transcription)
            if not match:
                continue
            transcription = match.group(1) 
            wordlist_for_language.append({"ord":word, "IPA":transcription, "språk":lang})
        data.close()
        
        # Vi blander sammen ordene, og reduserer mengder hvis listen er for lang
        random.shuffle(wordlist_for_language)
        wordlist_for_language = wordlist_for_language[:max_nb_words_per_language]
        
        full_wordlist += wordlist_for_language
        print("ferdig!")

    # Nå bygger vi en DataFrame med alle ordene
    full_wordlist = pandas.DataFrame.from_records(full_wordlist)
        
    # Og vi blander sammen ordene i tilfeldig rekkefølge
    full_wordlist = full_wordlist.sample(frac=1)
    
    return full_wordlist

#####################################
# METODER FOR ENTITETSGJENKJENNING
#####################################
                  

def preprocess(tagged_text):
    """Tar en tokenisert tekst med XML tags (som f.eks. <ORG>Stortinget</ORG>) og
    returnerer en liste over setninger (som selv er lister over tokens), sammen med
    en liste av samme lengde som inneholder de markerte navngitte enhetene. """
    
    sentences = []
    spans = []
    
    for i, line in enumerate(tagged_text.split("\n")):

        tokens = []
        spans_in_sentence = []
        
        for j, token in enumerate(line.split(" ")):
            
            # Hvis token starter med en XML tag
            start_match = re.match("<(\w+?)>", token)
            if start_match:
                new_span = (j, None, start_match.group(1))
                spans_in_sentence.append(new_span)
                token = token[start_match.end(0):]
            
            # Hvis token slutter med en XML tag
            end_match = re.match("(.+)</(\w+?)>$", token)
            if end_match:
                if not spans_in_sentence or spans_in_sentence[-1][1]!=None:
                    raise RuntimeError("Closing tag without corresponding open tag")
                start, _ , tag = spans_in_sentence[-1]
                if tag != end_match.group(2):
                    raise RuntimeError("Closing tag does not correspond to open tag")
                token = token[:end_match.end(1)]
                spans_in_sentence[-1] = (start, j+1, tag)
                
            tokens.append(token)
            
        sentences.append(tokens)
        spans.append(spans_in_sentence)
        
    return sentences, spans


def get_spans(label_sequence):
    """Gitt en labelsekvens med BIO markering, returner en lister over "spans" med 
    navngitte enheter. Metoden er altså den motsatte av get_BIO_sequence"""
    
    spans = []           
    i = 0
    while i < len(label_sequence):
        label = label_sequence[i]
        if label.startswith("B-"):
            start = i
            label = label[2:]
            end = start + 1
            while end < len(label_sequence) and label_sequence[end].startswith("I-%s"%label):
                end += 1
            spans.append((start, end, label))
            i = end
        else:
            i += 1
    return spans


def postprocess(sentences, spans):
    """Gitt en liste over setninger og en tilsvarende liste over "spans" med
    navngitte enheter, produserer en tekst med XML markering."""
    
    tagged_sentences = []
    for i, sentence in enumerate(sentences):
        new_sentence = list(sentence)
        for start, end, tag in spans[i]:
            new_sentence[start] = "<%s>%s"%(tag, new_sentence[start])
            new_sentence[end-1] = "%s</%s>"%(new_sentence[end-1], tag)
        tagged_sentences.append(" ".join(new_sentence))
     
    return "\n".join(tagged_sentences)

        
    
