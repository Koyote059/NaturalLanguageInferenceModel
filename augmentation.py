import random
import re
from abc import ABC
from typing import Any, Tuple, List, Dict

import spacy
from datasets import load_dataset
from nltk.corpus import wordnet


# TODO could use NN to transform verbs in correct tense


def substitute(record: dict, sentence: str, span: Tuple[int, int] | int):
    if isinstance(span, int):
        span = (span, span)
    tokens: List[str] = [element['rawText'] for element in record['srl']['hypothesis']['tokens']]
    del tokens[span[0]:span[1] + 1]
    tokens.insert(span[0], sentence)
    record['hypothesis'] = ' '.join(tokens)
    record['hypothesis'] = re.sub(r'\s+([,\.!?])', r'\1', record['hypothesis'])


def parse_propositions(record) -> dict[Any, dict[str, dict[Any, Any] | Any]]:
    """
    Given a record and a set of wanted roles, it parses the semantics of each role.
    :param record: the record from which to extract the information.
    :return: a dict with the following structure:
       'frameName' ( from englishPropbank ): {
            'pos': POS from wordnet ( Like NOUN etc... ),
            'text': text from the sentence,
            'index': index of the word in the sentence,
            'lemma': the base lemma of the text,
            'synset': the wordnet synset ( can be 'O' if unique )
            'text': Verb Text
            'aux': { // The auxiliary verb, optional ( like was, is, have... )
                'pos': POS from wordnet ( Like NOUN etc... ),
                'text': text from the sentence,
                'index': index of the word in the sentence,
                'lemma': the base lemma of the text,
                'synset': the wordnet synset ( can be 'O' if unique )
                'text': Verb Text
            },
            adv: {
                // Same thing as aux for adverbs
            }
            'roles': {
                ARG0: [{
                    'pos': POS from wordnet ( Like NOUN etc... ),
                    'text': text from the sentence,
                    'index': index of the word in the sentence,
                    'lemma': the base lemma of the text,
                    'synset': the wordnet synset ( can be 'O' if unique )
                    }, {
                        ...
                    }]
                },
                ARG1: {
                    ...
                },
                ... ( all the roles from englishProbanks)
            }

        }
    """
    parsing = dict()
    semantics_annotations = record['srl']['hypothesis']["annotations"]
    wsd = record['wsd']['hypothesis']
    for verb_srl in semantics_annotations:
        verb_index = verb_srl['tokenIndex']
        english_propbank = verb_srl['englishPropbank']
        frame_name = english_propbank['frameName']
        roles = english_propbank['roles']
        parsing[frame_name] = {
            "pos": wsd[verb_index]['pos'],
            "text": wsd[verb_index]['text'],
            'index': wsd[verb_index]['index'],
            'lemma': wsd[verb_index]['lemma'],
            'synset': wsd[verb_index]['nltkSynset'],
            'roles': {}
        }
        aux_verb = [word for word in wsd if word["pos"] == 'AUX']
        if len(aux_verb) > 0:
            aux = aux_verb[0]
            if aux['index'] != parsing[frame_name]['index']:
                parsing[frame_name]['aux'] = {
                    "pos": aux['pos'],
                    "text": aux['text'],
                    "index": aux['index'],
                    "lemma": aux['lemma'],
                    "synset": aux['nltkSynset'],
                }

        adv_verbs = [word for word in wsd if word["pos"] == 'ADV']
        if len(adv_verbs) > 0:
            adv = adv_verbs[0]
            parsing[frame_name]['adv'] = {
                "pos": adv['pos'],
                "text": adv['text'],
                "index": adv['index'],
                "lemma": adv['lemma'],
                "synset": adv['nltkSynset'],
            }
        for role in roles:
            role_type = role['role']
            span = role['span']

            parsing[frame_name]['roles'][role_type] = [
                {"pos": wsd[index]['pos'], "text": wsd[index]['text'], 'index': wsd[index]['index'],
                 'lemma': wsd[index]['lemma'], 'synset': wsd[index]['nltkSynset']} for index in
                range(span[0], span[1] + 1) if len(wsd) > index
            ]

    return parsing


class AugmentationMethod(ABC):

    def __call__(self, record: dict):
        return self.augment(record)

    def augment(self, record: dict):
        pass


class RandomHypothesisSubstitution(AugmentationMethod):

    # TODO careful! This creates a lot of neutrals
    #   Ideas:
    #       - Balance dataset
    #       - Don't use it in records who already are Neutral

    def __init__(self, dataset):
        self.dataset = dataset

    def augment(self, record: dict):
        random_record = random.choice(self.dataset)
        while record['id'] == random_record['id']:
            random_record = random.choice(self.dataset)

        new_record = record.copy()
        new_record['id'] = record["id"] + str(len(self.dataset))
        new_record['hypothesis'] = random_record['hypothesis']
        new_record['label'] = 'NEUTRAL'
        new_record['wsd']['hypothesis'] = random_record['wsd']['hypothesis']
        new_record['srl']['hypothesis'] = random_record['srl']['hypothesis']

        return new_record


class HypernymySubstitution(AugmentationMethod):

    @staticmethod
    def _get_noun_to_augment(record):
        proposition_semantics = parse_propositions(record)
        if len(proposition_semantics) == 0:
            return None
        proposition = random.choice(list(proposition_semantics.keys()))
        proposition_semantics = proposition_semantics[proposition]
        semantic_senses = ['ARG0', 'ARG2', 'ARG1']
        for sense in semantic_senses:
            if sense not in proposition_semantics['roles']:
                continue
            arg1 = proposition_semantics['roles'][sense]
            nouns = [word for word in arg1 if word["pos"] == 'NOUN']
            if len(nouns) == 0:
                continue
            noun = random.choice(nouns)
            return noun
        return None

    def augment(self, record: dict):
        noun = self._get_noun_to_augment(record)
        if noun is None or len(noun["text"]) == 0 or noun["text"][0].isupper():
            return None

        if noun['synset'] == 'O':
            synset = wordnet.synsets(noun['lemma'])[0]
        else:
            synset = wordnet.synset(noun['synset'])
        hypernims = synset.hypernyms()
        if len(hypernims) == 0:
            return None
        hypernym = hypernims[0].lemmas()[0].name()
        if "_" in hypernym:
            hypernym = ' '.join(hypernym.split("_"))
        new_record = record.copy()
        substitute(new_record, hypernym, (noun['index'], noun['index']))
        return new_record


class VerbNegation(AugmentationMethod):
    # Can improve by using an actual dictionary
    @staticmethod
    def invert_verb(verb: str, aux: str | None):

        if aux is None:
            if verb == 'is':
                return "is not"
            if verb == "isn't":
                return "is"
            if verb == 'are':
                return "are not"
            if verb == 'have':
                return "have not"
            if verb == 'has':
                return "has not"
            if verb == 'were':
                return "were not"
            if verb == 'was':
                return "was not"
            if verb.endswith("ed"):
                return f"didn't {verb[:-2]}"
            if verb.endswith("'t"):
                return f"did {verb}"
            if verb.endswith("s"):
                return f"doesn't {verb[:-1]}"

            return f"not {verb}"
        else:
            if aux == "didn't":
                return f"{verb}ed"
            if aux == 'was':
                return f"wasn't {verb}"
            if aux == "wasn't":
                return f"was {verb}"
            if aux == 'has':
                return f"has {verb}"
            elif aux == 'is':
                return f"is not {verb}"
            else:
                return f"not {verb}"

    def augment(self, record: dict) -> dict | None:
        proposition_semantics = parse_propositions(record)
        if len(proposition_semantics) == 0:
            return None
        proposition = random.choice(list(proposition_semantics.keys()))
        proposition_semantic = proposition_semantics[proposition]
        verb = proposition_semantic['text']
        aux = proposition_semantic['aux']['text'] if 'aux' in proposition_semantic else None
        inverted_verb = self.invert_verb(verb, aux)
        if inverted_verb is None:
            return None
        new_record = record.copy()
        if aux is None:
            substitute(new_record, inverted_verb, (proposition_semantic['index'], proposition_semantic['index']))
        else:
            verb_index = proposition_semantic['index']
            aux_index = proposition_semantic['aux']['index']
            if (verb_index - aux_index) > 1 or (verb_index - aux_index) < -1:
                return None
            substitute(new_record, inverted_verb, (min(verb_index, aux_index), max(verb_index, aux_index)))

        if new_record['label'] == 'CONTRADICTION':
            new_record['label'] = 'ENTAILMENT'
        elif new_record['label'] == 'ENTAILMENT':
            new_record['label'] = 'CONTRADICTION'
        return new_record


class DateSubstitution(AugmentationMethod):

    @staticmethod
    def _randomly_alter_number(number):
        if ',' in number:
            number = number.replace(",", ".")
        if number.endswith("s"):
            number = number[:-1]
        try:
            was_int = True
            number = int(number)
        except ValueError:
            was_int = False
            try:
                number = float(number)
            except ValueError:
                return None

        offset = random.uniform(-0.2, 0.1)
        transformed_number = number + offset * number
        if was_int:
            transformed_number = int(transformed_number)
        return str(transformed_number)

    def augment(self, record: dict):
        proposition_semantics = parse_propositions(record)
        if len(proposition_semantics) == 0:
            return None
        proposition = random.choice(list(proposition_semantics.keys()))
        proposition_semantic = proposition_semantics[proposition]
        if "ARGM-TMP" not in proposition_semantic["roles"]:
            return None
        temporal_semantic = proposition_semantic["roles"]["ARGM-TMP"]
        numbers = [semantic for semantic in temporal_semantic if semantic['pos'] == 'NUM']
        if len(numbers) == 0:
            return None
        number = random.choice(numbers)
        altered_number = self._randomly_alter_number(number["text"])
        if altered_number is None:
            return None
        new_record = record.copy()
        substitute(new_record, altered_number, number["index"])
        if new_record['label'] == 'CONTRADICTION':
            new_record['label'] = 'ENTAILMENT'
        elif new_record['label'] == 'ENTAILMENT':
            new_record['label'] = 'CONTRADICTION'
        return new_record


class AdverbInversion(AugmentationMethod):

    @staticmethod
    def _invert_adverb(adverb: dict) -> str | None:
        if adverb["synset"] == "O":
            synset = wordnet.synsets(adverb["lemma"])
            if synset is None:
                return None
            synset = synset[0]
        else:
            synset = wordnet.synset(adverb["synset"])
        antonyms = set()
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
        if len(antonyms) == 0:
            return None
        return random.choice(list(antonyms))

    def augment(self, record: dict) -> dict | None:
        proposition_semantics = parse_propositions(record)
        if len(proposition_semantics) == 0:
            return None
        proposition = random.choice(list(proposition_semantics.keys()))
        proposition_semantic = proposition_semantics[proposition]
        if 'ARG1' not in proposition_semantic['roles']:
            return None
        patient = proposition_semantic['roles']['ARG1']
        adverbs = [elem for elem in patient if elem['pos'] == 'ADV']
        if len(adverbs) == 0:
            return None
        random_adverb = random.choice(adverbs)
        inverted_adverb = self._invert_adverb(random_adverb)
        if inverted_adverb is None:
            return None
        new_record = record.copy()
        substitute(new_record, inverted_adverb, random_adverb["index"])
        if new_record['label'] == 'CONTRADICTION':
            new_record['label'] = 'ENTAILMENT'
        if new_record['label'] == 'ENTAILMENT':
            new_record['label'] = 'CONTRADICTION'
        return new_record


class VerbSynonymSubstitution(AugmentationMethod):
    nlp = spacy.load("en_core_web_sm")

    # python -m spacy download en_core_web_sm important!

    @staticmethod
    def conjugate_verb(verb: str, new_verb: str):
        if verb.endswith("ed"):
            return f"{new_verb}ed"
        if verb.endswith("'t"):
            return f"{new_verb}n't"
        if verb.endswith("s"):
            return f"{new_verb}s"
        return new_verb

    @staticmethod
    def _find_synonym(verb: dict) -> str | None:
        if verb["synset"] == "O":
            synset = wordnet.synsets(verb["lemma"])
            synset = [s for s in synset if s.pos() == 'v']

            if synset is None or len(synset) == 0:
                return None
            synset = synset[0]
        else:
            synset = wordnet.synset(verb["synset"])
        synonyms = set()
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().lower())
        if verb["lemma"].lower() in synonyms:
            synonyms.remove(verb["lemma"].lower())
        if 'beryllium' in synonyms:
            synonyms.remove('beryllium')
        if len(synonyms) == 0:
            return None
        random_verb = random.choice(list(synonyms)).replace("_", " ")
        return VerbSynonymSubstitution.conjugate_verb(verb=verb["text"], new_verb=random_verb)

    def augment(self, record: dict):
        proposition_semantics = parse_propositions(record)
        if len(proposition_semantics) == 0:
            return None
        proposition = random.choice(list(proposition_semantics.keys()))
        proposition_semantic = proposition_semantics[proposition]
        inverted_verb = self._find_synonym(proposition_semantic)
        if inverted_verb is None:
            return None
        new_record = record.copy()
        substitute(new_record, inverted_verb, proposition_semantic['index'])
        return new_record


class AugmentationPipeline:

    def __init__(self, augmentation_methods: Dict[AugmentationMethod, float]):
        self.augmentation_methods = augmentation_methods

    def choose_methods(self) -> List[AugmentationMethod]:
        chosen_methods = []
        keys = list(self.augmentation_methods.keys())
        probabilities = list(self.augmentation_methods.values())
        while keys:
            method = random.choices(keys, probabilities, k=1)[0]
            index = keys.index(method)
            chosen_methods.append(method)
            del keys[index]
            del probabilities[index]

        return chosen_methods

    def __call__(self, record):
        return self.augment(record)

    def augment(self, record: dict) -> dict | None:
        methods = self.choose_methods()
        for method in methods:
            print("Trying method: ", method.__class__.__name__)
            augmented_record = method.augment(record)
            if augmented_record:
                return augmented_record
        return None

