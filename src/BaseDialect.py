from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict, Counter
import pickle as pkl
import pandas as pd
import numpy as np
import json, re, os, spacy, nltk

import lemminflect
import random
import string
from nltk.corpus import wordnet as wn
from collections import Counter
import inflect
from .inflect.english import English
from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn
from nltk.metrics.distance import edit_distance
from nltk.stem.wordnet import WordNetLemmatizer
import neuralcoref


class BaseDialect(object):
    def __init__(self, lexical_swaps={}, morphosyntax=True, seed=None):
        self.string = ""
        self.rules = defaultdict(dict)
        self.executed_rules = {}
        self.doc = None
        self.tokens = []
        self.end_idx = 0
        self.morphosyntax = morphosyntax
        self.modification_counter = Counter()
        self.lexical_swaps = lexical_swaps

        # Note(will): For some reason, I was seeing Seg Faults when calling self.nlp
        # if I did neuralcoref.add_to_pipe(spacy.load(self.nlp)) after load
        self.nlp = spacy.load("en_core_web_sm")
        neuralcoref.add_to_pipe(spacy.load("en_core_web_sm"))
        self.inflector = English()
        self.inflection = inflect.engine()
        self.cmudict = cmudict.dict()
        self.wnl = WordNetLemmatizer()
        
        if seed:
            self.set_seed(seed)
        
        # variables
        self.aint = False

        # pulled from https://github.com/ddemszky/textbook-analysis/tree/master/wordlists
        self.HUMAN_NOUNS = set(
            pd.read_csv(
                "resources/people_terms.csv", names=["noun", "race/gender", "category"]
            )["noun"].values
        )

        # pulled from http://simple.wiktionary.org/wiki/Category%3aUncountable_nouns
        # and https://gist.github.com/sudodoki/b5408fa4ba752cc22597250fc58a5970/raw/13e4ed5f8367982c8fbdc9c7ddffc8c5c008bf9d/nouns.txt
        with open("resources/mass_nouns.txt", "r") as infile:
            self.MASS_NOUNS = set([x.strip() for x in infile.readlines()])

        # universal dependency names for objects
        self.OBJECTS = {"dobj", "iobj", "obj", "pobj", "obl", "attr"}

        self.MODALS = {'has to', 'were able to', 'was able to', 'could', 'shall', 'should', 'ought to', 'might', 'can', 'must', 'may', 'would', 'is able to', 'will', 'am able to'}
        
        self.NOUN_MOD_DEPS = {'amod', 'npmod', 'appos', 'det', 'tmod', 'nummod', 'nmod', 'neg', 'poss', 'compound'}
        
        # https://archiewahwah.wordpress.com/speech-verbs-list/
        # https://blog.abaenglish.com/communication-verbs-in-english-talk-speak-tell-or-say/
        self.SPEAK_VERBS = {'announce', 'answer', 'claim', 'confirm', 'confess', 'convey', 'convince', 'comment', 'declare', 'discuss', 'exclaim', 'explain', 'mention', 'pronounce', 'reply', 'report', 'reveal', 'say', 'scream', 'shout', 'speak', 'state', 'tell', 'yell'}
        
        self.NEGATIVES = self.load_dict('resources/negatives.json')
        self.NEGATIVES_NO_MORE = self.load_dict('resources/negatives_no_more.json')
        self.REFLEXIVES_SUBJ = self.load_dict('resources/reflexives_subj.json')
        self.REFLEXIVES_NUMBER_SWAP = self.load_dict('resources/reflexives_number_swap.json')
        self.AAVE_REFLEXIVES = self.load_dict('resources/reflexives_aave.json')
        self.EMPHATIC_REFLEXIVES = self.load_dict('resources/reflexives_emphatic.json')
        self.REFLEXIVES_REGULARIZED = self.load_dict('resources/reflexives_regularized.json')
        self.REFLEXIVES_OBJ_PRON = self.load_dict('resources/reflexives_obj_pron.json')
        self.PAST_MODAL_MAPPING = self.load_dict('resources/modals_past.json')
        self.POSSESSIVES = self.load_dict('resources/possessives.json')
        self.PLURAL_DETERMINERS = self.load_dict('resources/det_plural.json')
        self.POSSESSIVE_OBJ_DET = self.load_dict('resources/det_possessive_obj.json')
        self.PRONOUN_OBJ_TO_SUBJ = self.load_dict('resources/pron_obj_to_subj.json')
        self.FLAT_ADV_ADJ = self.load_dict('resources/adv_adj_flat.json')
        self.NOUN_CONCRETENESS = self.load_dict('resources/noun_concreteness.json')
        
        # https://www.ef.com/wwen/english-resources/english-grammar/adverbs-degree/
        self.DEGREE_MODIFIER_ADV_ADJ = self.load_dict('resources/adv_adj_degree_modifier.json')
           
        # this list was drawn from VerbNet and compiled with `build_benefactive_ditransitive_verb_list.py`
        # with verb_transitivity.tsv from https://github.com/wilcoxeg/verb_transitivity/raw/master/verb_transitivity.tsv
        with open('resources/benefactive_verbs.txt', 'r') as infile:
            self.DITRANSITIVE_VERBS_BENEFACTIVE = {lemma.strip() for lemma in infile.readlines()}
            
        # this list was drawn from VerbNet and compiled with `build_benefactive_ditransitive_verb_list.py`
        with open('resources/ditransitive_dobj_verbs.txt', 'r') as infile:
            self.DITRANSITIVE_VERBS = {lemma.strip() for lemma in infile.readlines()}
            
        # this list was drawn from VerbNet and compiled with `build_benefactive_ditransitive_verb_list.py`
        with open('resources/transitive_dobj_verbs.txt', 'r') as infile:    
            self.TRANSITIVE_VERBS = {lemma.strip() for lemma in infile.readlines()}
            

    def __getstate__(self):
        return {"dialect": self.dialect_name}

    def __setstate__(self):
        return self

    def __hash__(self):
        return hash(self.dialect_name)

    def __str__(self):
        return self.dialect_name
    
    def set_seed(self, seed):
        random.seed(seed)
    
    def load_dict(self, fn):
        with open(fn, 'r') as infile:
            return json.load(infile)

    def surface_sub(self, string):
        """Cleaning up the mess left by the morphosyntactic transformations (analogous to surface-structure in Chomsky's surface/deep structure distinction)"""
        for method in self.surface_subs:
            string = method(string)
        return string.strip()

    def transform(self, string):
        return self.convert_sae_to_dialect(string)
    
    def convert_sae_to_dialect(self, string):
        """Full Conversion Pipeline"""
        self.update(string)
        for feature_name in self.morphosyntax_transforms.keys():
            method = getattr(self, feature_name)
            method()

        transformed = self.surface_sub(self.compile_from_rules())
        return (
            self.capitalize(transformed) if self.is_capitalized(string) else transformed
        )

    def clear(self):
        """clear all memory of previous string"""
        self.string = ""
        self.rules = defaultdict(dict)
        self.doc = None
        self.tokens = []
        self.end_idx = 0
        self.modification_counter = {label: 0 for label in list(self.morphosyntax_transforms.keys()) + ['lexical', 'total']}

    def update(self, string):
        """update memory to the current string"""
        self.clear()
        self.string = string
        self.doc = self.nlp(string)
        self.tokens = list(self.doc)
        self.end_idx = len(string) - 1

    def set_rule(self, token, value, function_name, check_capital=True, token_alignment={}):
        if not len(function_name):
            print(3/0) # Throw error
        
        if check_capital and token.text[0].isupper() and len(value):
            remainder = ""
            if len(value)>1:
                remainder = value[1:]
            value = value[0].capitalize() + remainder
            
        start_idx = token.idx
        end_idx = min(token.idx + len(token) - 1, self.end_idx)
        if (not len(value)) \
        and (end_idx + 1 <= self.end_idx) and str(self.doc)[end_idx+1] == " " \
        and ((start_idx==0) or (str(self.doc)[start_idx-1]==" ")):
            # deletion should include the following space
            end_idx += 1

        span = (
            start_idx,  # starting with this token
            end_idx # and ending right before the next
        )
        
        self.rules[function_name][span] = {
            'value': value,
            'type': function_name,
            'token_alignment': token_alignment
        }

        if function_name:
            self.modification_counter[function_name] += 1
            self.modification_counter["total"] += 1
            
    def set_rule_span(self, span, value, function_name, check_capital=True, token_alignment={}):
        text = str(self.doc)
        if check_capital and text[span[0]].isupper() and len(value):
            remainder = ""
            if len(value)>1:
                remainder = value[1:]
            value = value[0].capitalize() + remainder
            
        start_idx, end_idx = span
        end_idx = min(end_idx-1, self.end_idx)
        if (not len(value)) and (end_idx + 1 <= self.end_idx) and str(self.doc)[end_idx+1] == " ":
            end_idx += 1
        
        self.rules[function_name][(start_idx, end_idx)] = {
            'value': value,
            'type': function_name,
            'token_alignment': token_alignment
        }

        if function_name:
            self.modification_counter[function_name] += 1
            self.modification_counter["total"] += 1

    def get_full_word(self, token):
        """spaCy tokenizes [don't] into [do, n't], and this method can be used to recover the full word don't given either do or n't"""
        start = 0
        end = self.end_idx
        for idx in range(token.idx, -1, -1):
            if self.string[idx] not in string.ascii_letters + "'’-":
                start = idx + 1
                break
        for idx in range(token.idx + 1, self.end_idx + 1):
            if self.string[idx] not in string.ascii_letters + "'’-":
                end = idx
                break
        return self.string[start:end]

    def is_contraction(self, token):
        """returns boolean value indicating whether @token has a contraction"""
        for idx in range(token.idx, self.end_idx + 1):
            if self.string[idx] in ["'", "’"]:
                return True
            if re.match("^[\w-]+$", self.string[idx]) is None:
                return False
        return False

    def is_negated(self, token):
        """returns boolean value indicating whether @token is negated"""
        return any([c.dep_ == "neg" for c in token.children])

    def capitalize(self, string):
        """takes a word array @string and capitalizes only the first word in the sentence"""
        if len(string):
            return string[0].capitalize() + string[1:]
        return ""

    def is_capitalized(self, string):
        """returns boolean value indicating whether @string is capitalized"""
        return self.capitalize(string) == string

    def is_modal(self, token):
        """returns boolean value indicating whether @token is a modal verb"""
        if token.tag_ == "MD":
            return True

        start_idx = token.idx
        for m in self.MODALS:
            end_idx = start_idx + len(m)
            if self.string[start_idx:end_idx] == m:
                return True
        return False

    def has_object(self, token):
        """returns boolean value indicating whether @token has an object dependant (child)"""
        return any([c.dep_ in self.OBJECTS for c in token.children])
    
    def has_aux(self, verb):
        """returns boolean value indicating whether @verb has an aux dependant (child)"""
        return  any([c.dep_ == 'aux' for c in verb.children])

    def adjective_synsets(self, synsets):
        """returns a list of all adjectives in the WordNet vocabulary"""
        return [s for s in synsets if s.pos() in {"a"}]

    def is_gradable_adjective(self, word: str):
        """returns boolean value indicating whether @word is a gradable adjective"""
        orig_synsets = set(self.adjective_synsets(wn.synsets(word)))

        superlative = {}
        superlative[0] = word + "est"  # green -> greenest
        superlative[1] = word + word[-1] + "est"  # hot -> hottest
        if len(word) and word[-1] == "e":
            superlative[2] = word[:-1] + "er"  # 'blue' -> 'bluest'
        if len(word) > 1 and word[-2] == "y":
            superlative[3] = word[:-1] + "ier"  # happy -> happiest

        return any(
            [
                len(
                    set(self.adjective_synsets(wn.synsets(sup))).intersection(
                        orig_synsets
                    )
                )
                for sup in superlative.values()
            ]
        )
                        
    def she_inanimate_objects(self):
        # feature 1
        self.inanimate_objects(p=0, name="she_inanimate_objects")
        
    def he_inanimate_objects(self):
        # feature 2
        self.inanimate_objects(p=1, name="he_inanimate_objects")
        
    def inanimate_objects(self, p=0.5, name="inanimate_objects"):
        if self.doc._.coref_clusters:
            for cluster in self.doc._.coref_clusters:
                human_span = False
                prp = None
                for span in cluster.mentions:
                    for token_idx in range(span.start, span.end):
                        token = self.doc[token_idx]
                        if self.token_is_human(token):
                            human_span = True
                        if token.tag_ == 'PRP':
                            prp = token
                if prp and (not human_span) and prp.text not in {'she', 'him'}:
                    if 'subj' in prp.dep_:
                        replace = "he" if random.random() < p else 'she'
                        self.set_rule(prp, replace, name)
                    else:
                        replace = "him" if random.random() < p else 'her'
                        self.set_rule(prp, replace, name)
                        
    def referential_thing(self):
        """Adds "the thing" for referential (non-dummy) "it"""
        # feature 3
        replace = "the thing"
        for token in self.tokens:
            if token.dep_ != "expl" and token.lower_=='it':
                self.set_rule(token, replace, "referential_thing")
    
    def pleonastic_that(self, p=0.5):
        """Adds the existential "that" / "thass" construction"""
        # feature 4, (32)
        replace = "that's" if random.random() < p else "thass"
        for token in self.tokens:
            if token.dep_ == "expl" and token.lower_ == 'it':
                self.set_rule(token, replace, "pleonastic_that")
                self.set_rule(token.head, " ", "pleonastic_that") 
                
    def em_subj_pronoun(self):
        # feature 5
        for token in self.tokens:
            if (token.dep_ == 'nsubj' or self.is_coordinate_subject_subordinate(token)) and token.lower_ in {'he', 'she', 'it'}:
                self.set_rule(token, "em", "em_subj_pronoun")
                
    def em_obj_pronoun(self):
        # feature 6
        for token in self.tokens:
            if 'obj' in token.dep_ and token.lower_ in {'him', 'it'}:
                self.set_rule(token, "em", "em_obj_pronoun")
    
    def me_coordinate_subjects(self):
        # feature 7
        self.coordinate_subjects(substitute="me", name="me_coordinate_subjects")
        
    def myself_coordinate_subjects(self):
        # feature 8
        self.coordinate_subjects(substitute="myself", name="myself_coordinate_subjects")
    
    def coordinate_subjects(self, substitute="me", name="coordinate_subjects"):
        for token in self.tokens:
            if token.lower_ == 'i':
                if self.is_coordinate_subject_subordinate(token):
                    self.set_rule(token, "", name)
                    self.set_rule(token.head, f"{substitute} and {self.build_transposable_NP_limited(token.head, name)}", name)
                    for c in token.head.children:
                        if c.dep_ == 'cc':
                            self.set_rule(c, "", name)
                elif self.is_coordinate_subject_superordinate(token):
                    conj = None
                    cc = None
                    for c in token.children:
                        if c.dep_ == 'cc':
                            cc = c
                        if c.dep_ == 'conj':
                            conj = c
                    if conj and cc:
                        self.set_rule(token, f"{substitute} {cc} {self.build_transposable_NP_limited(conj, name)}", name)
                        self.set_rule(cc, "", name)
                        self.set_rule(conj, "", name)
                        
                        
    def is_coordinate_subject(self, token):
        """Returns True if the token belongs to a subject conjunction"""
        return self.is_coordinate_subject_subordinate(token) or self.is_coordinate_subject_superordinate(token)
    
    def is_coordinate_subject_subordinate(self, token):
        return (token.dep_=='conj' and token.head.dep_ == 'nsubj')
    
    def is_coordinate_subject_superordinate(self, token):
        return (token.dep_ == 'nsubj' and 'cc' in {c.dep_ for c in token.children}) 
        
    def benefactive_dative(self):
        # feature 9
        for token in self.tokens:
            if token.lemma_ in self.DITRANSITIVE_VERBS_BENEFACTIVE:
                subj = self.get_subj_matrix(token)
                dobjs = [c for c in token.children if c.dep_=='dobj']
                children = {c.text for c in token.children}
                children_deps = self.get_children_deps(token)
                # we are looking for benefactive verbs that don't already have that THETA role filled
                # we have to exclude cases of xcomp children because the spaCy parser *frequently* 
                # misunderstands the beneficiary to be the subject of a clausal complement
                if subj and (subj.text=='I') and ('prep' not in children_deps) and ('dative' not in children_deps) and ('ccomp' not in children_deps) and (len(dobjs)<=1):
                    self.set_rule(token, f"{token.text} me", 'benefactive_dative')
                
                
    def no_gender_distinction(self):
        # feature 10
        mapping = {
            ('she', 'nsubj'): 'he',
            ('he', 'nsubj'): 'she',
            ('her', 'dobj'): 'him',
            ('him', 'dobj'): 'her',
            ('her', 'pobj'): 'him',
            ('him', 'pobj'): 'her',
            ('her', 'poss'): 'his',
            ('his', 'poss'): 'her',
            ('hers', 'attr'): 'his',
            ('his', 'attr'): 'hers',
        }
        
        for token in self.tokens:
            tup = (token.lower_, token.dep_)
            if tup in mapping:
                self.set_rule(token, mapping[tup], "no_gender_distinction")
        
    def regularized_reflexives(self):
        # feature 11
        for token in self.tokens:
            if token.lower_ in self.REFLEXIVES_REGULARIZED:
                self.set_rule(token, self.REFLEXIVES_REGULARIZED[token.lower_], 'regularized_reflexives')
                
    def regularized_reflexives_object_pronouns(self):
        # feature 12
        for token in self.tokens:
            if token.lower_ in self.REFLEXIVES_OBJ_PRON:
                self.set_rule(token, self.REFLEXIVES_OBJ_PRON[token.lower_], 'regularized_reflexives')
                
    def regularized_reflexives_aave(self):
        # feature 13
        for token in self.tokens:
            if token.lower_ in self.AAVE_REFLEXIVES:
                self.set_rule(token, self.AAVE_REFLEXIVES[token.lower_], 'regularized_reflexives')
                
    def reflex_number(self):
        # feature 14
        for token in self.tokens:
            if token.lower_ in self.REFLEXIVES_NUMBER_SWAP:
                self.set_rule(
                    token, self.REFLEXIVES_NUMBER_SWAP[token.lower_], "reflex_number"
                )
                
    def absolute_reflex(self):
        # feature 15
        for token in self.tokens:
            if token.dep_ == 'nsubj' and token.lower_ in self.REFLEXIVES_SUBJ:
                self.set_rule(token, self.REFLEXIVES_SUBJ[token.lower_], "absolute_reflex")
                
    def emphatic_reflex(self):
        # feature 16
        for token in self.tokens:
            if token.lower_ in self.EMPHATIC_REFLEXIVES:
                self.set_rule(
                    token, self.EMPHATIC_REFLEXIVES[token.lower_], "emphatic_reflex"
                )
                
    def reflexives_swap(self):
        # features 11-16
        reflexives_combined = defaultdict(set)
        for dic in [self.REFLEXIVES_REGULARIZED, self.AAVE_REFLEXIVES, self.REFLEXIVES_NUMBER_SWAP, self.EMPHATIC_REFLEXIVES, self.REFLEXIVES_OBJ_PRON]:
            for key in dic:
                reflexives_combined[key].add(dic[key])
        
        for sae in reflexives_combined.keys():
            for match in re.finditer(r"\b%s\b" % sae, str(self.doc), re.IGNORECASE):
                span = match.span()

                swap = random.sample(list(reflexives_combined[sae]), 1)[0]
                self.set_rule_span(span, swap, "reflexives_swap")
                
    def pronoun_swap(self, swaps = {
            'my': {'I', 'me', 'mine'},
            'our': {'we', 'us', 'ourn', 'oursn', 'ourns'},
            'his': {'he', 'him', 'hisn'},
            'her': {'she', 'hersn'},
            'their': {'they', 'them', 'theirn'},
            'your': {'you', "y'all's", 'yourn'},
            'me': {'us', 'I'},
            'her': {'she'},
            'him': {'he'},
            'them': {'they'},
            'I': {'me'},
            'she': {'her'},
            'he': {'him'},
            'us': {'we'},
            'them': {'they'},
            'you': {'ye'}
        }, name="pronoun_swap"):
        # features 18-27; 29-31; 33; 35
        for sae in swaps.keys():
            for match in re.finditer(r"\b%s\b" % sae, str(self.doc), re.IGNORECASE):
                span = match.span()

                swap = random.sample(list(swaps[sae]), 1)[0]
                self.set_rule_span(span, swap, name)
    
    def my_i(self):
        # feature 18
        return self.pronoun_swap(swaps = {'my': {'I'}}, name='my_i')
        
    def our_we(self):
        # feature 19
        return self.pronoun_swap(swaps = {'our': {'we'}}, name='our_we')
        
    def his_he(self):
        # feature 20
        return self.pronoun_swap(swaps = {'his': {'he'}}, name='his_he')
        
    def their_they(self):
        # feature 21
        return self.pronoun_swap(swaps = {'their': {'they'}}, name='their_they')
        
    def your_you(self):
        # feature 22
        return self.pronoun_swap(swaps = {'your': {'you'}}, name='your_you')
        
    def your_yalls(self):
        # feature 23
        return self.pronoun_swap(swaps = {'your': {"y'all's"}}, name='your_yalls')
        
    def his_him(self):
        # feature 24
        return self.pronoun_swap(swaps = {'his': {"him"}}, name='his_him')
        
    def their_them(self):
        # feature 25
        return self.pronoun_swap(swaps = {'their': {"them"}}, name='their_them')
        
    def my_me(self):
        # feature 26
        return self.pronoun_swap(swaps = {'my': {"me"}}, name='my_me')
        
    def our_us(self):
        # feature 27
        return self.pronoun_swap(swaps = {'our': {"us"}}, name='our_us')
        
    def me_us(self):
        # feature 29
        return self.pronoun_swap(swaps = {'me': {"us"}}, name='me_us')
        
    def non_coordinated_subj_obj(self):
        # feature 30
        swaps = {
            'me': 'I',
            'her': 'she',
            'him': 'he',
            'them': 'they',
            'us': 'we'
        }
        for token in self.tokens:
            if 'obj' in token.dep_ and token.lower_ in swaps:
                self.set_rule(token, swaps[token.lower_], 'non_coordinated_subj_obj')
                
    def non_coordinated_obj_subj(self):
        # feature 31
        swaps = {
            'I': 'me',
            'she': 'her',
            'he': 'him',
            'they': 'them',
            'we': 'us'
        }
        for token in self.tokens:
            if 'subj' in token.dep_ and token.lower_ in swaps:
                self.set_rule(token, swaps[token.lower_], 'non_coordinated_obj_subj')
    
    def nasal_possessive_pron(self):
        # feature 33
        for token in self.tokens:
            if token.tag_ == 'PRP$':
                replace = token.text+'n'
                if replace == 'oursn':
                    replace = 'ourn'
                self.set_rule(token, replace, 'nasal_possessive_pron')
                
    def surface_contract(self, string):
        """Contract verbs and fix errors in contractions"""
        
        # fixes
        orig = string  # .copy()

        string = re.sub(r"\byou all\b", "y'all", string, flags=re.IGNORECASE)
        string = re.sub(r"\bnn't\b", "n't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bwilln't\b", "won't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bI'm going to\b", "I'ma", string, flags=re.IGNORECASE)
        string = re.sub(r"\bI am going to\b", "I'ma", string, flags=re.IGNORECASE)
        string = re.sub(r"\bI will\b ", "I'ma ", string, flags=re.IGNORECASE)
        string = re.sub(r"\bhave to\b", "gotta", string, flags=re.IGNORECASE)
        string = re.sub(r"\bneed to\b", "gotta", string, flags=re.IGNORECASE)
        string = re.sub(r"\bgot to\b", r"gotta", string, flags=re.IGNORECASE)
        string = re.sub(r"\btrying to\b", "tryna", string, flags=re.IGNORECASE)
        string = re.sub(r"\bbeen been\b", "been", string, flags=re.IGNORECASE)

        if string != orig:
            self.modification_counter["lexical"] += 1
            self.modification_counter["total"] += 1

        return string
    
    def yall(self):
        # feature 34
        return self.pronoun_swap(swaps = {'you': {"y'all"}}, name='yall')
        
    def you_ye(self):
        # feature 35
        return self.pronoun_swap(swaps = {'you': {"ye"}}, name='you_ye')
                
    def interrogatives(self, p=0.5):
        for token in self.tokens:
            if token.tag_ in {"WP", "WP$"} and self.is_clause_initial(token):
                if random.random() < p:
                    self.set_rule(
                        token,
                        token.text + "-" + token.text.lower(),
                        "reduplicate_interrogative",
                    )
                else:
                    self.set_rule(
                        token,
                        token.text + "-all",
                        "plural_interrogative",
                    )
    def plural_interrogative(self):
        # feature 39
        self.interrogatives(0.0)
                
    def reduplicate_interrogative(self):
        # feature 40
        self.interrogatives(1.0)
                
    def anaphoric_it(self):
        # feature 41
        if self.doc._.coref_clusters:
            for cluster in self.doc._.coref_clusters:
                for span in cluster.mentions:
                    for token_idx in range(span.start, span.end):
                        token = self.doc[token_idx]
                        if token.lower_ == "they":
                            self.set_rule(token, "it", "anaphoric_it")
                            
    def object_pronoun_drop(self):
        # feature 42
        for token in self.tokens:
            if token.lemma_ in {"it", "them", "-PRON-"} and "dobj" in token.dep_:
                self.set_rule(token, "", "object_pronoun_drop")
                
    def null_referential_pronouns(self):
        # feature 43
        if self.doc._.coref_clusters:
            for cluster in self.doc._.coref_clusters:
                for span in cluster.mentions:
                    for token_idx in range(span.start, span.end):
                        token = self.doc[token_idx]
                        if token.tag_ == "PRP" and "subj" in token.dep_:
                            self.set_rule(token, "", "null_referential_pronouns")
                            
    def it_dobj(self):
        # feature 45
        for token in self.tokens:
            if token.pos_=='VERB' and token.dep_=='advcl' and token.lemma_ in self.TRANSITIVE_VERBS:
                children = {c.dep_ for c in token.children}
                if 'mark' in children and 'dobj' not in children and 'prep' not in children:
                    self.set_rule(token, f"{token} it", "it_dobj")
                        
    def get_referential_it_tokens(self):
        referential_it_tokens = []
        if self.doc._.coref_clusters:
            for cluster in self.doc._.coref_clusters:
                for span in cluster.mentions:
                    for token_idx in range(span.start, span.end):
                        token = self.doc[token_idx]
                        if token.lower_ == 'it' and token.head.lower_ == 'is':
                            referential_it_tokens.append(token)
        return referential_it_tokens
                            
    def it_is_referential(self):
        # feature 46
        referential_it_tokens = self.get_referential_it_tokens()
        for token in self.tokens:
            if token.lower_ == 'it' and token.head.lower_ == 'is':
                if token in referential_it_tokens:
                    self.set_rule(token, "", "it_is_referential")
                    
    def it_is_non_referential(self):
        # feature 47
        referential_it_tokens = self.get_referential_it_tokens()
        for token in self.tokens:
            if token.lower_ == 'it' and token.head.lower_ == 'is':
                if token not in referential_it_tokens:
                    self.set_rule(token, "", "it_is_non_referential")
    
    def regularized_plurals(self):
        # feature (48), 49
        for token in self.tokens:
            if (
                token.tag_ == "NNS"
            ):  
                lemma = self.inflector.singularize(token.text)
                if not lemma:
                    #print(token.text, 'NNS but not plural?')
                    continue
                elif not len(lemma)>1:
                    continue
                elif lemma[-1] == 'y':
                    continue # no need to change the [y + s --> ies] orthographic convention
                elif lemma[-2:] =='es':
                    continue # shouldn't mess with these sense the suffix is already plural
                if lemma[-1] in {"s", "x", "z"} or lemma[-2:] in {'sh', 'ch'}:
                    if lemma + "es" != token.text:
                        self.set_rule(token, lemma + "es", "regularized_plurals")
                else:
                    if lemma + "s" != token.text:
                        self.set_rule(token, lemma + "s", "regularized_plurals")
                        
    def plural_preposed(self, preposition='alla'):
        for token in self.tokens:
            if token.tag_ == "NNS":
                if not any([c.dep_ in self.NOUN_MOD_DEPS for c in token.children]):
                    lemma = self.inflector.singularize(token.text)
                    if lemma and lemma!=token.lower_:
                        self.set_rule(token, f"{preposition} {lemma}", "plural_preposed")
                        
    def plural_postposed(self, postposition='them'):
        for token in self.tokens:
            if token.tag_ == "NNS":
                lemma = self.inflector.singularize(token.text)
                if not any([ (c.dep_ in self.NOUN_MOD_DEPS) and (c.i > token.i) for c in token.children]):
                    if lemma and lemma!=token.lower_:
                        self.set_rule(token, f"{lemma} {postposition}", "plural_postposed")
                        
    def plural_suffix(self, text):
        if text[-1] in {"s", "x", "z"} or text[-2:] in {'sh', 'ch'}:
            return text + "es"
        return text + "s"
                        
    def mass_noun_plurals(self):
        # feature 55
        # BUG: overapplies to group nouns like "two sisters in law" --> "two sisters in laws"
        for token in self.tokens:
            if token.tag_ == "NN" and token.lower_ in self.MASS_NOUNS:
                lemma = self.inflector.singularize(token.text)
                if lemma[-1] in {"s", "x", "z"}:
                    self.set_rule(token, lemma + "es", "mass_noun_plurals")
                else:
                    self.set_rule(token, lemma + "s", "mass_noun_plurals")
                self.subj_sentence_agreement(token, "mass_noun_plurals", form_num=1)
                        
    def zero_plural_after_quantifier(self):
        # feature 56
        for token in self.tokens:
            if token.tag_ == "NNS" and "nummod" in [c.dep_ for c in token.children]:
                self.set_rule(
                    token, self.inflector.singularize(token.text), "zero_plural_after_quantifier"
                )
                
    def plural_to_singular_human(self):
        # feature 57
        for token in self.tokens:
            if self.token_is_human(token):
                replace = self.inflector.singularize(token.text)
                if replace != token.text:
                    self.set_rule(
                        token,
                        self.inflector.singularize(token.text),
                        "plural_to_singular_human",
                    )
                    
    def zero_plural(self):
        # feature 58
        for token in self.tokens:
            if token.tag_ == "NNS" and token.lower_ not in self.HUMAN_NOUNS:
                replace = self.inflector.singularize(token.text)
                if replace != token.text:
                    self.set_rule(
                        token,
                        replace,
                        "zero_plural",
                    )
                    
    def get_det_pobj(self, token):
        det = None
        prep = None
        pobj = None
        for c in token.children:
            if c.dep_=='det':
                det = c
            elif c.dep_=='prep':
                prep = c
                for child in c.children:
                    if child.dep_=='pobj':
                        pobj = child
        return det, prep, pobj
                    
    def double_determiners(self):
        # feature 59
        for token in self.tokens:
            if token.pos_ == "NOUN":
                det, prep, pobj = self.get_det_pobj(token)
                if pobj and pobj.lower_ in self.POSSESSIVE_OBJ_DET:
                    double_det = self.POSSESSIVE_OBJ_DET[pobj.lower_]
                    if det:
                        if det.lower_ in {'this', 'that', 'these', 'those'}: # demonstratives
                            self.set_rule(det, f"{det.text} {double_det}", "double_determiners")
                            self.set_rule(prep, "", "double_determiners")
                            self.set_rule(pobj, "", "double_determiners")
                    else: # no determiner so add one before the transposable NP
                        transposable_NP = self.build_transposable_NP_limited(token, "double_determiners")
                        self.set_rule(token, f"{double_det} {transposable_NP}", "double_determiners")
                        self.set_rule(prep, "", "double_determiners")
                        self.set_rule(pobj, "", "double_determiners")
                    
    def definite_for_indefinite_articles(self):
        # feature 60
        replace = "the" 
        for token in self.tokens:
            if token.tag_ == "DT" and token.text.lower() == "a":
                self.set_rule(token, replace, "definite_for_indefinite_articles")
                
    def indefinite_for_definite_articles(self):
        # feature 61
        replace = "a" 
        for token in self.tokens:
            if token.tag_ == "DT" and token.text.lower() == "the":
                self.set_rule(token, replace, "indefinite_for_definite_articles")
                
    def remove_determiners(self, token, name="remove_determiners"):
        """Convert 'the dog' into just 'dog'"""
        for child in token.children:
            if child.dep_ == "det":
                self.set_rule(child, "", name)
                
    def remove_det(self, which={'a', 'an', 'the'}, name='remove_det'):
        for token in self.tokens:
            if token.pos_ == "DET" and token.lemma_ in which and (not self.noun_is_modified(token.head)):
                self.set_rule(token, "", name)  
                
    def remove_det_definite(self):
        # feature 62
        self.remove_det(which={'the'}, name='remove_det_definite')
            
    def remove_det_indefinite(self):
        # feature 63
        self.remove_det(which={'a', 'an'}, name='remove_det_indefinite')
                
    def remove_or_indefinite_for_definite_articles(self, p=0.5):
        # (features 61, 62, 63)
        tag, replace = "indefinite_articles", "a" 
        if random.random() < p:
            tag, replace = "remove_det", ""
        for token in self.tokens:
            if token.tag_ == "DT" and token.text.lower() == "the":
                self.set_rule(token, replace, tag)
                
    def definite_abstract(self, concrete_max=2.5):
        # feature 64
        for token in self.tokens:
            if (
                (token.tag_ == "NN") #or (token.tag_ == "NNP" and not self.is_capitalized(token.text))
                and token.lower_ in self.NOUN_CONCRETENESS
                and self.NOUN_CONCRETENESS[token.lower_] <= concrete_max
                and not any([c.dep_ in self.NOUN_MOD_DEPS for c in token.children])
                #and not any([c.dep_ in {'pobj'} for c in token.head.children])
            ):
                self.set_rule(token, "the " + token.text, "definite_abstract")
                
    def indefinite_for_zero(self):
        # feature 65
        for token in self.tokens:
            if (
                (token.tag_ == "NN") or (token.tag_ == "NNP" and not self.is_capitalized(token.text))
                and not any([c.dep_ in self.NOUN_MOD_DEPS for c in token.children])
                and not any([c.dep_ in {'pobj'} for c in token.head.children])
            ):
                self.set_rule(token, "a " + token.text, "indefinite_for_zero") 
                
    def definite_ORG(self):
        for token in self.tokens:
            if (
                token.tag_ == "NNP"
                and token.dep_ != 'compound'
                and token.ent_type_ == "ORG"
                and "DT" not in [c.tag_ for c in token.children]
                and "compound" not in [c.dep_ for c in token.children]
            ):
                self.set_rule(token, "the " + token.text, "definite_ORG")
                
    def indef_one(self):
        # feature 66
        for token in self.tokens:
            if token.lower_ == "a" and token.pos_ == "DET":
                self.set_rule(token, "one", "indef_one")
                    
    def demonstrative_for_definite_articles(self):
        # feature 67
        for token in self.tokens:
            if (
                token.tag_ == "DT"
                and token.text.lower() == "the"
                and token.head.tag_ in {"NN", "NNP"}
                #and random.random() < p
            ):
                demonstrative = "that" if token.head.tag_ == "NN" else "those"
                self.set_rule(token, demonstrative, "demonstrative_for_definite_articles")
                
    def those_them(self):
        # feature 68
        for match in re.finditer(r"\bthose\b", str(self.doc), re.IGNORECASE):
            span = match.span()

            swap = "them"
            self.set_rule_span(span, swap, "those_them")
    
    def proximal_distal_demonstratives(self):
        # feature 70
        for token in self.tokens:
            if token.dep_ == 'det':
                if token.lower_ in {'this', 'these'}:
                    self.set_rule(token, f"{token.text} here", "proximal_distal_demonstratives")
                elif token.lower_ in {'that', 'those'}:
                    self.set_rule(token, f"{token.text} there", "proximal_distal_demonstratives")
    
    def demonstrative_no_number(self):
        # feature 71
        mapping = {'these': 'this',
                   'those': 'that'}
        for token in self.tokens:
            if token.lower_ in mapping:
                self.set_rule(token, mapping[token.lower_], "demonstrative_no_number")
                
    def existential_possessives(self):
        # feature 73
        for token in self.tokens:
            if token.dep_ == 'dobj' and 'relcl' not in {c.dep_ for c in token.children}:
                head = token.head
                subj = None
                for c in head.children:
                    if c.dep_ == 'nsubj':
                        subj = c
                if head.dep_ == 'ROOT' and head.lower_ == 'have' and subj and subj.text=='I':
                    transposable_NP = self.build_transposable_NP_limited(token, "existential_possessives")
                    copula = 'are' if (token.dep_ in {'NNS', 'NNPS'}) or (token.lower_ in self.MASS_NOUNS) else 'is'
                    self.set_rule(subj, f"{transposable_NP} {copula} there", "existential_possessives")
                    self.set_rule(head, "", "existential_possessives")
                    self.set_rule(token, "", "existential_possessives")
                
                
    def possessives(self, belong=False, before=False, name='possessives'):
        # feature 74, 75, 76
        poss = {
            'my': 'me',
            'your': 'you',
            'his': 'him',
            'their': 'them',
            'our': 'us'
        }
        for token in self.tokens:
            if token.dep_ == 'poss' and not token.head.dep_=='poss':
                for child in token.children:
                    if child.dep_ == 'case':
                        self.set_rule(child, '', name)
                possessor = self.build_transposable_NP_limited(token,
                                                               name=name,
                                                               deps={'compound', 'clf', 'amod', 'nummod', 'poss'}
                                                              )
                
                if possessor in poss:
                    possessor = poss[possessor]
                
                possessed = self.build_transposable_NP_limited(token.head, 
                                                               name=name,
                                                               deps={'compound', 'amod', 'nummod'}
                                                              )
                
                self.set_rule(token.head, "", name)
                if belong:
                    self.set_rule(token, f"{possessor} belong {possessed}", name)
                elif before:
                    self.set_rule(token, f"for {possessor} {possessed}", name)
                else:
                    self.set_rule(token, f"the {possessed} for {possessor}", name)
        
    def possessives_for_post(self):
        # feature 74
        self.possessives(belong=False, before=False, name='possessives_for_post')
        
    def possessives_for_pre(self):
        # feature 75
        self.possessives(belong=False, before=True, name='possessives_for_pre')
        
    def possessives_belong(self):
        # feature 76
        self.possessives(belong=True, before=False, name='possessives_belong')
    
    def build_transposable_NP_limited(self, token, name, deps={'compound', 'case', 'clf', 'amod', 'nummod', 'poss'}):
        components = [token]
        for child in token.children:
            if (child.tag_ == 'DT') or (child.dep_ in deps):
                components.append(child)
                self.set_rule(child, "", name)
        components = [t.lower_ for t in sorted(components, key=lambda t: t.i)]
        return " ".join(components)
    
    # TODO: this should NOT apply where there is ellipsis of the object as in:
    # "I borrowed Fred's diagram of a snake's eye because Steve's had been stolen."
    # ==> "I borrowed Fred diagram of a snake's eye because Steve been stolen."
    def null_genitive(self):
        """Removes the possessive s and other possessive morphology"""
        
        # feature 77
        for token in self.tokens:
            if token.tag_ == "POS":
                self.set_rule(token, "", "null_genitive")  # drop and leave a space
                
    def double_comparative(self):
        # feature 78
        for token in self.tokens:
            # we don't want to act on graded quantifiers
            if token.tag_ == "JJR" and token.lower_ not in {'more', 'much', 'many', 
                                                            'few', 'fewer', 'little', 'less'}:
                self.set_rule(token, "more " + token.text, "double_comparative")
                for c in token.head.children:
                    if c.tag_ == 'DT' and c.lemma_ == 'an':
                        # determiner should match "more"
                        self.set_rule(c, "a", "double_comparative")
                        
    def double_superlative(self):
        # feature 78
        self.analytic_or_double_superlative(p=0.0)
                        
    def synthetic_superlative(self):
        # feature 79
        for token in self.tokens:
            if token.tag_ == 'JJ':
                for c in token.children:
                    if c.tag_ == 'RBS' and c.lemma_ == 'most' and token.text[:-3] != 'est' and not self.has_form(token.text, 'v'):
                        base = token.text
                        if base[-1] == 'e':
                            base = base[:-1]
                        if base[-1] == 'y':
                            base = base[:-1] + 'i'    
                        self.set_rule(c, "", "synthetic_superlative")
                        self.set_rule(token, base + 'est', "synthetic_superlative")
        
    def analytic_superlative(self):
        # feature 80
        self.analytic_or_double_superlative(p=1.0)
                
    def analytic_or_double_superlative(self, p=0.5):
        for token in self.tokens:
            if token.tag_ == "JJS" and token.lower_ not in {'most', 'fewest', 'least'}:
                decision = random.random()
                replace = "most " + token.lemma_ if decision < p else "most " + token.text
                transformation_name = "analytic_superlative" if decision < p else "double_superlative"
                self.set_rule(
                    token,
                    replace,
                    transformation_name
                )
                for c in token.head.children:
                    if c.tag_ == 'DT' and c.lemma_ == 'an':
                        # determiner should match "most"
                        self.set_rule(c, "a", transformation_name)
                        
    def more_much(self):
        # feature 81
        for match in re.finditer(r"\bmore\b", str(self.doc), re.IGNORECASE):
            span = match.span()

            swap = "much"
            self.set_rule_span(span, swap, "more_much")
    
    def comparative_as_to(self, p=0.5):
        # feature 82
        for token in self.tokens:
            if token.lower_ == 'than': # not quantmod (e.g. "two is more than one")
                if token.head.tag_ == "JJR":
                    replace = 'as' if random.random() < p else "to"
                    self.set_rule(token, replace, 'comparative_as_to')
                
    def comparative_than(self):
        # feature 84
        for token in self.tokens:
            if token.lower_ == 'than' and token.dep_ == 'prep': # not quantmod (e.g. "two is more than one")
                if token.head.tag_ in {"RBR", "JJR"} and token.head.lower_ == 'more':
                    self.set_rule(token.head, "", 'comparative_than')
                else:
                    for child in token.head.children:
                        if child.lower_ == 'more':
                            self.set_rule(child, "", 'comparative_than')
                
                
    def comparative_more_and(self):
        # feature 85
        for token in self.tokens:
            if token.lower_ == 'than' and token.dep_ == 'prep': # not quantmod (e.g. "two is more than one")
                if token.head.tag_ in {"RBR", "JJR"} and token.head.lower_ == 'more':
                    self.set_rule(token, "and", 'comparative_more_and')
                else:
                    for child in token.head.children:
                        if child.lower_ == 'more':
                            self.set_rule(token, "and", 'comparative_more_and')
                
    def zero_degree(self):
        # feature 86
        for token in self.tokens:
            if token.tag_ in {"RBS", "JJS"} and token.lower_ == 'most':
                self.set_rule(token, "", 'zero_degree')
                
    def adj_postfix(self):
        # feature 87
        for token in self.tokens:
            if token.tag_ == 'JJ' and token.head.pos_ == 'NOUN' and not any([c.dep_ in {'compound', 'case', 'clf', 'nummod', 'poss'} for c in token.head.children]):
                adj = self.build_transposable_conj(token, 'adj_postfix')
                self.set_rule(token, "", 'adj_postfix')
                self.set_rule(token.head, f"{token.head.text} {adj}", 'adj_postfix')
                
    def build_transposable_conj(self, token, name):
            """Return the string form of @token with all of compound structure, and also clear out
            the dependents in the rule set so that @token is "transposable" or movable throughout the
            rule set
            """
            components = [token]
            for child in token.children:
                if child.dep_ in {'cc', 'conj'}:
                    components.append(child)
                    self.set_rule(child, "", name)
            components = [t.text for t in sorted(components, key=lambda t: t.i)]
            return " ".join(components)
        
    def progressives_no_swap_do(self):
        self.progressives(swap_do_aux=False)
                
    def progressives(self, swap_do_aux=True):
        # feature 88
        for token in self.tokens:
            modify = False
            swap_aux = None
            if token.lemma_ not in {'be', 'do', 'have', 'get'}:
                if token.tag_ in {'VBP', 'VBZ'} and not self.has_aux(token): # simple present
                    modify = True
                elif token.tag_ == 'VB':
                    modify = True
                    for c in token.children:
                        if c.dep_ == 'aux':
                            if c.lemma_ == 'do' and swap_do_aux:
                                swap_aux = c
                            else:
                                modify = False

                if modify:
                    num = self.get_verb_number(token)
                    cop = "is" 
                    if num==0:
                        cop = 'am'
                    elif num==2:
                        cop = "are"

                    if swap_aux:
                        self.set_rule(swap_aux, cop, 'progressives')
                        self.set_rule(token, token._.inflect("VBG"), 'progressives')
                    else:
                        self.set_rule(token, f"{cop} {token._.inflect('VBG')}", 'progressives')
                        
    def standing_stood(self):
        # feature 95
        for token in self.tokens:
            if token.lower_ == 'standing' and token.pos_=='VERB' and 'nsubj' in {c.dep_ for c in token.children}:
                self.set_rule(token, 'stood', 'standing_stood')
    
    def resultative_past_participle(self, require_expletive=False, name="resultative_past_participle"):
        for token in self.tokens:
            if token.dep_ == 'relcl' and token.tag_ == 'VBD':
                subj = None
                for c in token.children:
                    if c.dep_ == 'nsubj' and c.lower_ in {'who', 'that'}:
                        subj = c
                if subj and ((not require_expletive) or ('expl' in {c.dep_ for c in token.head.head.children})):
                    self.set_rule(token, token._.inflect("VBN"), name)
                    self.set_rule(subj, "", name) 
    
    def that_resultative_past_participle(self, name="that_resultative_past_participle"):
        # feature 96
        self.resultative_past_participle(require_expletive=True, name=name)
        
    def medial_object_perfect(self):
        # feature 97
        for token in self.tokens:
            if token.tag_ == 'VBN' and not self.has_additional_aux(token):
                has = None
                subj = None
                obj = None
                mod = None
                for child in token.children:
                    if child.dep_=='aux' and child.lower_ in {'have', 'has', "'ve", "'s"}:
                        has = child
                    elif child.dep_ == 'nsubj':
                        subj = child
                    elif 'mod' in child.dep_:
                        mod = child
                if has and subj and (not mod):
                    for child in token.children:
                        if child.dep_ == 'dobj':
                            obj = self.build_transposable_NP(child, 'medial_object_perfect', clear_components=True)
                if has and obj:
                    self.set_rule(token, f"{obj} {token.lower_}", "medial_object_perfect")
                        
    def after_perfect(self):
        # feature 98
        for token in self.tokens:
            if token.tag_ == 'VBN' and not self.has_additional_aux(token):
                aux = None
                nsubj = None
                mod = None
                for child in token.children:
                    if 'mod' in child.dep_:
                        mod = child#self.set_rule(child, "", "after_perfect")
                    elif child.dep_ == 'aux':
                        aux = child
                    elif child.dep_ == 'nsubj':
                        nsubj = child
                if nsubj and aux:
                    copula = 'are' if (nsubj.dep_ in {'NNS', 'NNPS'}) or (nsubj.lower_ in self.MASS_NOUNS) else 'is'
                    self.set_rule(token, f'after {token._.inflect("VBG")}', "after_perfect")
                    self.set_rule(aux, copula, "after_perfect")
                    if mod:
                        self.set_rule(mod, "", "after_perfect")
                
    def simple_past_for_present_perfect(self):
        # feature 99
        self.simple_past_or_present_for_present_perfect(p=1.0)
        
    def simple_past_or_present_for_present_perfect(self, ever=False, p=0.5):
        for token in self.tokens:
            change_token = False
            decision = random.random()
            name = "simple_past_for_present_perfect" if decision < p else "present_for_exp_perfect"
            
            if token.tag_ == 'VBN' and not self.has_additional_aux(token):
                for child in token.children:
                    if child.dep_ == 'aux' and child.lower_ in {'have', 'has', "'ve", "'s"}:
                        replace = "ever" if ever else ""
                        self.set_rule(child, replace, name)
                        change_token = True
            if change_token:
                num = self.get_verb_number(token)
                pos = "VBD" if decision < p else "VBP"
                if pos == 'VBP' and num == 1:
                    # account for 3rd person singular form, different from others
                    pos = 'VBZ'
                replace = token._.inflect(pos, form_num=num)
                self.set_rule(token, replace, name)
        
    def present_perfect_for_past(self):
        # feature 100
        for token in self.tokens:
            if token.tag_ == "VBD" and (token.dep_!='aux') and not self.has_aux(token) and not len(set(self.get_full_word(token)).intersection({"'", "’"})):
                num = self.get_verb_number(token)
                have = "had" if num == 1 else "have"
                replace = "{} {}".format(have, token._.inflect("VBN", form_num=num))
                self.set_rule(token, replace, "present_perfect_for_past")
        
    def present_for_exp_perfect(self):
        # feature 101
        # change continuative or experential past to simple present
        self.simple_past_or_present_for_present_perfect(p=0.0)
        
    def be_perfect(self, progressive=False):
        # feature 102
        for token in self.tokens:
            if token.tag_ == 'VBN' and not self.has_additional_aux(token):
                aux = None
                nsubj = None
                mod = None
                for child in token.children:
                    if 'mod' in child.dep_:
                        mod = child
                    elif child.dep_ == 'aux':
                        aux = child
                    elif child.dep_ == 'nsubj':
                        nsubj = child
                if nsubj and aux:
                    copula = 'are' if (nsubj.dep_ in {'NNS', 'NNPS'}) or (nsubj.lower_ in self.MASS_NOUNS) or (nsubj.lower_ in {'we', 'they'}) else 'is'
                    self.set_rule(aux, copula, "be_perfect")
                    if progressive:
                        self.set_rule(token, f'{token._.inflect("VBG")}', "be_perfect")
                    if mod:
                        self.set_rule(mod, "", "be_perfect")
                        
    def do_tense_marker(self):
        # feature 103
        for token in self.tokens:
            if token.dep_=='ROOT' and token.pos_=='VERB':
                c_deps = {c.dep_ for c in token.children}
                if not len({'aux', 'cop', 'mark', 'advmod', 'expl', 'vocative'}.intersection(c_deps)):
                    
                    do = None
                    verb_number = self.get_verb_number(token)
                    if token.tag_ in {'VBZ', 'VBP', 'VB'}:
                        if verb_number==1:
                            do = 'does'
                        else:
                            do = 'do'
                    elif token.tag_ == 'VBD':
                        do = 'did'
                    if do:
                        self.set_rule(token, f"{do} {token._.inflect('VB')}", "do_tense_marker")
                
    def completive_been_done(self,drop_aux=False, name="completive_done"):
        """Implements completive done"""
        for token in self.tokens:
            if token.lower_ in {"done", "been"}:
                # if the verb is done/been, we don't want to duplicate it
                continue
            if token.tag_ == "VBD":
                if any(
                    [
                        child.dep_ == "npadvmod" or child.lemma_ in {"already", "ago"}
                        for child in token.children
                    ]
                ):
                    self.set_rule(token, f"done {token.text}", name)
            elif token.tag_ == "VBN":
                for child in token.children:
                    if (
                        child.dep_ == "aux"
                        and child.lemma_ == "have"
                        and not self.is_contraction(child)
                    ):
                        replace = "done" if drop_aux else f"{child} done"
                        self.set_rule(child, replace, name)
        
    def completive_done(self):
        # feature 104
        self.completive_been_done(drop_aux=True, name="completive_done")
        
    def completive_have_done(self):
        # feature 105
        self.completive_been_done(drop_aux=False, name="completive_have_done")
                        
    def irrealis_be_done(self):
        # feature 106
        for token in self.tokens:
            if (
                token.dep_ == "mark"
                and token.lower_ == "if"
                and token.head.dep_ == "advcl"
            ):
                # add 'be done' before ROOT in subordinate clause (token.head.head) and remove any other auxiliaries
                self.set_rule(
                    token.head.head,
                    "be done " + token.head.head.text,
                    "irrealis_be_done",
                )
                for c in token.head.head.children:
                    if c.dep_ == "aux":
                        self.set_rule(c, "", "irrealis_be_done")
                        
    def perfect_alternative(self, alternative, name):
        for token in self.tokens:
            if token.tag_ == 'VBN' and token.lemma_!='be' and not self.has_additional_aux(token):
                subj = self.get_subj(token)
                        
                for child in token.children:
                    if child.lower_ in {'have', 'has'}:# and random.random() < p:
                        num = self.get_verb_number(token)
                        replace = token._.inflect("VBD", form_num=num)
                        
                        replace_aux = ""
                        if (not subj) or (child.i < subj.i):
                            # subject aux inversion indicates a question and the necessity of "do support"
                            replace_aux = 'did'
                            replace = token.head._.inflect('VB', form_num=0)
                            
                        self.set_rule(child, replace_aux, name)
                        self.set_rule(token.head, f"{alternative} {replace}", name)        
                        
    def perfect_slam(self):
        # feature 107
        self.perfect_alternative("slam", 'perfect_slam')
        
    def present_perfect_ever(self):
        # feature 108
        self.simple_past_or_present_for_present_perfect(ever=True, p=0.0)
                        
    def perfect_already(self):
        # feature 109
        self.perfect_alternative("already", 'perfect_already')
                        
    def completive_finish(self):
        # feature 110
        for token in self.tokens:
            change_token = False
            if token.tag_ == 'VBN' and not self.has_additional_aux(token):
                for child in token.children:
                    if child.dep_ == 'aux' and child.lower_ in {'have', 'has', "'ve", "'s"}:
                        replace = "finish"
                        if child.lower_ in {"'ve", "'s"}:
                            replace = " finish"
                        self.set_rule(child, replace, "completive_finish")
                        change_token = True
            if change_token:
                replace = token._.inflect("VB", form_num=0)
                self.set_rule(token, replace, "completive_finish")
                
    def past_been(self):
        # feature 111
        for token in self.tokens:
            if token.dep_=='ROOT' and token.tag_ == 'VBD':
                c_deps = {c.dep_ for c in token.children}
                if not len({'aux', 'cop', 'mark', 'advmod', 'expl', 'vocative'}.intersection(c_deps)):
                    self.set_rule(token, f"been {token.lower_}", "past_been")
                    
    def bare_perfect(self):
        # feature 112
        for token in self.tokens:
            if token.tag_ == 'VBN' and not self.has_additional_aux(token):
                self.set_rule(token, token._.inflect("VB"), "bare_perfect")
                
    def future_sub_gon(self):
        # feature 114
        return self.future(replace="gon", remove_to=True, remove_head_copula=True, will_add_phrasal=False, name="future_sub_gon")
    
    def volition_changes(self, replace="waan"):
        # feature 115
        for token in self.tokens:
            if token.lower_=='want':
                for child in token.children:
                    if child.dep_ in {'xcomp', 'ccomp'}:
                        for c in child.children:
                            if c.tag_=='TO':
                                self.set_rule(c, '', 'volition_changes')
                                self.set_rule(token, replace, 'volition_changes')
                                
    def future(self, replace='coming', remove_to=False, remove_head_copula=False, will_add_phrasal=True, name='come_future'):
        for token in self.tokens:
            if token.lower_ in {'going', 'about'}:
                changed = False
                for child in token.children:
                    if child.dep_ in {'xcomp', 'ccomp'}:
                        for c in child.children:
                            if c.tag_=='TO':
                                if remove_to:
                                    self.set_rule(c, '', name)
                                self.set_rule(token, replace, name)
                                changed = True
                if changed and remove_head_copula:
                    if token.head.lemma_ == 'be':
                        self.set_rule(token.head, "", name)
                    else:
                        for c in token.children:
                            if c.dep_=='aux' and c.lemma_ == 'be':
                                self.set_rule(c, "", name)
            if token.lower_ == "will" and token.tag_ == 'MD':
                for child in token.head.children:
                    if child.dep_ == 'nsubj' and child.i < token.i:
                        copula = 'are' if (child.dep_ in {'NNS', 'NNPS'}) or (child.lower_ in self.MASS_NOUNS) else 'is'
                        replace_ = replace
                        if will_add_phrasal:
                            replace_ = f"{copula} {replace_} to"
                        self.set_rule(token, replace_, name)
                                
    def come_future(self):
        # feature 116
        self.future(replace='coming', remove_to=False, remove_head_copula=False, will_add_phrasal=True, name='come_future')
                
    def present_for_neutral_future(self):
        # feature 117
        self.future(replace='', remove_to=True, remove_head_copula=True, will_add_phrasal=False, name='present_for_neutral_future')
        
    def is_am_1s(self):
        # feature 118
        for token in self.tokens:
            subj = None
            if token.lemma_ == 'be':
                for c in token.children:
                    if c.dep_ == 'nsubj':
                        subj = c
                for c in token.head.children:
                    if c.dep_ == 'nsubj':
                        subj = c
            elif token.lemma_ == 'will':
                for c in token.head.children:
                    if c.dep_ == 'nsubj':
                        subj = c
            if subj and subj.lower_ == 'i':
                if "'" in token.lower_:
                    self.set_rule(token, "'s", 'is_am_1s')
                else:
                    self.set_rule(token, 'is', 'is_am_1s')
                
    def will_would(self):
        # feature 119
        for token in self.tokens:
            # this will apply to questions as well
            if token.lower_ == "will" and token.tag_ == 'MD':
                self.set_rule(token, "would", "will_would")
                
    def if_would(self):
        # feature 120
        for token in self.tokens:
            if token.dep_ == 'mark' and token.lower_ == 'if':
                head = token.head
                if 'aux' not in {c.dep_ for c in head.children}:
                    self.set_rule(head, f"would {head._.inflect('VB')}", "if_would")
                
    def double_modals(self):
        # feature 121
        for token in self.tokens:
            if self.is_modal(token) and token.lemma_!= 'might':
                replace = "might "
                if "'" in token.lower_:
                    replace = " might "
                self.set_rule(token, replace + token.lemma_, "double_modals")
                
    def present_modals(self):
        # feature 123
        for token in self.tokens:
            if token.tag_ == 'MD' and token.lower_ in self.PAST_MODAL_MAPPING.keys():
                self.set_rule(token, self.PAST_MODAL_MAPPING[token.lower_], "present_modals")
                
    def finna_future(self):
        # feature 126
        self.future(replace='finna', remove_to=True, remove_head_copula=False, will_add_phrasal=False, name='finna_future')
        
    def fixin_future(self):
        # feature 126
        self.future(replace='fixin', remove_to=False, remove_head_copula=False, will_add_phrasal=True, name='fixin_future')
    
    def regularized_or_bare_past_tense(self, p_bare=0.5):
        # allows for a stochastic mixture between features 128 and 129
        for token in self.tokens:
            if token.tag_=='VBD' and token.lemma_ not in {'be', 'do', 'have'} and not self.has_aux(token):
                base = token._.inflect("VB", form_num=0)
                if random.random() < p_bare:
                    regularized = base
                    if regularized != token.text:
                        self.set_rule(token, regularized, "bare_past_tense")
                else:
                    regularized = base + 'ed'
                    if base[-1] == 'y':
                        # no need to mess with the y --> ied spelling pattern
                        continue
                    elif base[-1] == 'e':
                        regularized = base + 'd' 
                    elif self.double_consonant(base):
                        regularized = base + base[-1] + 'ed'
                    if regularized != token.text:
                        self.set_rule(token, regularized, "regularized_past_tense")
        
    def regularized_past_tense(self):
        # feature 128
        self.regularized_or_bare_past_tense(p_bare=0.0)
        
    def bare_past_tense(self):
        # feature 129
        self.regularized_or_bare_past_tense(p_bare=1.0)
        
    def past_for_past_participle(self):
        # feature 130
        for token in self.tokens:
            if token.tag_ == 'VBN' and token.lemma_!='be':
                replace = token._.inflect("VBD", form_num=0)
                if replace != token.text:
                    self.set_rule(token, replace, "past_for_past_participle")
        
    def participle_or_bare_past_tense(self, p_bare=0.5):
        # allows for a stochastic mixture between features 131, 132
        for token in self.tokens:
            if token.tag_=='VBD' and token.lemma_ not in {'be', 'do', 'have'} and not self.has_aux(token):
                if random.random() < p_bare:
                    regularized = token._.inflect("VB", form_num=0)
                    if regularized != token.text:
                        self.set_rule(token, regularized, "bare_past_tense")
                else:
                    regularized = token._.inflect("VBN", form_num=0)
                    if regularized != token.text:
                        self.set_rule(token, regularized, "participle_past_tense")
                        
    def participle_past_tense(self):
        # feature 131
        self.participle_or_bare_past_tense(p_bare=0)
        
    def bare_past_tense(self):
        # feature 132
        self.participle_or_bare_past_tense(p_bare=1)
        
    def double_past(self):
        # feature 133
        for token in self.tokens:
            if token.pos_ == 'VERB':
                for child in token.children:
                    if child.dep_ == 'aux' and child.tag_ == 'VBD':
                        self.set_rule(token, token._.inflect('VBD'), 'double_past')
                        
    def a_ing(self):
        # feature 134
        for token in self.tokens:
            if token.tag_ == 'VBG' and len(token)>3 and token.text[-3:] == 'ing':
                self.set_rule(token, f"a-{token.text}", "a_ing")
                
    def a_participle(self):
        # feature 135
        for token in self.tokens:
            if token.tag_ == 'VBN':
                self.set_rule(token, f"a-{token.text}", "a_participle") 
                
    def transitive_suffix(self, suffix="em"):
        # feature 143
        for token in self.tokens:
            if token.pos_ == 'VERB':
                c_deps = [c.dep_ for c in token.children]
                if 'dobj' in c_deps and 'iobj' not in c_deps and 'pobj' not in c_deps:
                    dobj = list(token.children)[c_deps.index('dobj')]
                    
                    noun_mod_deps = self.NOUN_MOD_DEPS.copy()
                    noun_mod_deps.remove('det')
                    
                    dobj_c_deps = {c.dep_ for c in dobj.children}
                    if not len(dobj_c_deps.intersection(set(noun_mod_deps))):
                        for c in dobj.children:
                            if c.dep_ == 'det':
                                self.set_rule(c, '', 'transitive_suffix')
                        self.set_rule(dobj, f"em {dobj.text}", "transitive_suffix")
                
    def got_gotten(self):
        # feature 145
        for token in self.tokens:
            if token.tag_ == 'VBN' and token.lower_ == "got":
                self.set_rule(token, "gotten", "got_gotten")
                
    def verbal_ing_suffix(self):
        # feature 146
        for token in self.tokens:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                if token.lower_[-3:]!='ing':
                    self.set_rule(token, token._.inflect('VBG'), "verbal_ing_suffix")
                
    def conditional_were_was(self, name="conditional_were_was"):
        # feature 147
        for token in self.tokens:
            if token.lower_ == "were" and ((token.dep_ == "advcl") or ("if" in {c.lower_ for c in token.children})):
                self.set_rule(token, "was", name)
                
    def serial_verb_give(self):
        # feature 148
        for token in self.tokens:
            if token.lower_ in {'for', 'to'} and token.dep_ == 'dative' and token.head.pos_ =='VERB' and self.has_concrete_object(token.head):
                self.set_rule(token, "give", "serial_verb_give")
                
    def serial_verb_go(self):
        # feature 149
        # TODO: filter only for verbs that indicate movement with this 'to' preposition
        for token in self.tokens:
            if token.lower_ in {'to'} and token.dep_ == 'prep' and token.head.pos_ =='VERB' and (token.head.lemma_ not in {'go', 'get', 'do', 'be'}) and (not self.has_prt_child(token)) and self.has_concrete_object(token):
                self.set_rule(token, "go", "serial_verb_go")
                
    def here_come(self):
        # feature 150
        for token in self.tokens:
            if (token.lower_ == 'here') and (token.head.lemma_!='be'):
                self.set_rule(token, "come", "here_come")
                
    def give_passive(self):
        # feature 153
        for token in self.tokens:
            if token.tag_ == 'VBN': # looking for passive construction
                subj = None
                pobj = None
                has_subj = False
                has_pobj = False
                
                for child in token.children:
                    if child.dep_ == 'nsubjpass':
                        has_subj = True
                    elif child.dep_ == 'agent':
                        for c in child.children:
                            if c.dep_ == 'pobj':
                                has_pobj = True
                                
                if has_subj and has_pobj:           
                    for child in token.children:
                        if child.dep_ == 'nsubjpass':
                            subj = self.build_transposable_NP(child, "give_passive")
                        elif child.dep_ == 'agent':
                            for c in child.children:
                                if c.dep_ == 'pobj':
                                    pobj = self.build_transposable_NP(c, "give_passive")
                                    
                    clause_origin = self.get_clause_origin(token)
                    left_edge = self.subtree_min_idx(token, use_doc_index=True)
                    left = " ".join([x.text for x in self.tokens[:left_edge+1] if x.text!= subj])
                    
                    self.set_rule(clause_origin, f"{left} {subj} give {pobj} {token._.inflect('VB', form_num=0)}", "give_passive")
                    self.clear_subtree(clause_origin, name="give_passive") 
                    
    
    def negative_concord(self, aint=False, name="negative_concord"):
        # feature 154
        aint = aint or self.aint
        for token in self.tokens:
            # if the token is a negation of a verbal head then
            if (
                (token.dep_ == "neg") or (token.lower_ in self.NEGATIVES.values())
            ) and (token.head.pos_ in {"VERB", "AUX"}):
                neg = ""
                for val in self.NEGATIVES.values():
                    if token.lower_ == val:
                        neg = val
                        break
                # consider special ain't for non-contractions
                if (token.head.lemma_ == "be") and not len(
                    set(self.get_full_word(token)).intersection({"'", "’"})
                ):
                    if aint:
                        self.set_rule(token.head, "ain't", name)
                    else:
                        self.set_rule(token.head, "isn't", name)
                    self.set_rule(token, neg, name)

                for child in token.head.children:
                    # Find the object child and check that it is an indefinite noun
                    if (child.dep_ in self.OBJECTS) and self.is_indefinite_noun(
                        child
                    ):  # \
                        # or ((child.dep_ in {'advmod'}) and (child.pos_ in {'ADV', 'ADJ'})):
                        if str(child) in list(self.NEGATIVES.keys()):
                            # there is a special NPI word that is the negation of this one
                            self.set_rule(
                                child, self.NEGATIVES[str(child)], name
                            )
                        else:
                            # otherwise, just append the prefix "no" to negate the word
                            self.set_rule(child, "no " + str(child), name)
                        self.remove_determiners(child, name)
                    # Next, look to the prepositional phrases and do the same thing
                    elif child.dep_ == "prep":
                        for c in child.children:
                            if (c.dep_ == "pobj") and self.is_indefinite_noun(c):
                                self.set_rule(c, "no " + str(c), name)
                                self.remove_determiners(c, name)
        
    def aint_neg(self, switch_lemmas={"be", "have", "do", "can"}, name="aint"):
        self.aint = True
        for token in self.tokens:
            if token.dep_=='neg':
                sub = None
                if token.head.lemma_ in switch_lemmas:
                    sub = token.head
                else:
                    for c in token.head.children:
                        if c.lemma_ in switch_lemmas:
                            sub = c
                if sub:
                    if "'" in token.text:
                        self.set_rule(sub, "ai", name)
                    else:    
                        self.set_rule(sub, "ain't", name)
                        self.set_rule(token, "", name)
        
    def aint_be(self):
        # feature 155
        self.aint_neg(switch_lemmas={"be"}, name="aint_be")
        
    def aint_have(self):
        # feature 156
        self.aint_neg(switch_lemmas={"have"}, name="aint_have")
                    
    def aint_before_main(self):
        # feature 157
        self.aint_neg(switch_lemmas={"do", "can"}, name="aint_before_main")
                        
    def dont(self, name='dont'):
        # feature 158
        for match in re.finditer(r"\bdoesn't\b", str(self.doc), re.IGNORECASE):
            self.set_rule_span(match.span(), "don't", name)
                        
    def never_negator(self, name="never_negator"):
        # feature 159
        for token in self.tokens:

            if ((token.dep_ == 'neg') and (token.head.tag_ == 'VB') and (token.head.lemma_ != 'be')):
                for c in token.head.children:
                    if c.lower_ == 'did':
                        
                        num = self.get_verb_number(token.head)
                        replace = token.head._.inflect("VBD", form_num=num)
                        
                        subj = self.get_subj(token.head)
                        if subj and c.i < subj.i: # question inversion
                            self.set_rule(token, ' ', name)
                            self.set_rule(token.head, f"never {replace}", name)
                        
                        else:
                            self.set_rule(c, '', name)
                            self.set_rule(token, 'never', name)
                            self.set_rule(token.head, f"{replace}", name)
                     
    def preverbal_negator(self, replace="no", name="no_preverbal_negator"):
        for token in self.tokens:
            if token.dep_=='neg':
                verb = token.head
                for c in verb.children:
                    if c.lemma_ == 'do':# and c.lower_ != 'do':
                        self.set_rule(c, "", name)
                        self.set_rule(token, replace, name)
                        if c.tag_=='VBD':
                            self.set_rule(verb, verb._.inflect("VBD"), name) 

    def no_preverbal_negator(self):
        # feature 160
        self.preverbal_negator(replace="no", name="no_preverbal_negator")
        
    def not_preverbal_negator(self):
        # feature 161
        self.preverbal_negator(replace="not", name="no_preverbal_negator")
        
    def nomo_existential(self, name="nomo_existential"):
        # feature 162
        for token in self.tokens:
            if token.dep_=='neg' and token.head.lemma_=='be':
                be = token.head
                expl = None
                any_ = None
                subj = None
                has_det = False
                has_mod = False
                for c in be.children:
                    if c.dep_=='expl' and c.lower_=='there':
                        expl = c
                    if c.dep_ in {'nsubj', 'attr'}:
                        subj = c
                    elif c.dep_ in {'advmod'}:
                        has_mod = True
                if subj:
                    for c in subj.children:
                        if c.dep_ == 'det':
                            if c.lower_ == 'any':
                                any_ = c
                            else:
                                has_det = True

                    if expl and not has_det and not has_mod:
                        self.set_rule(expl, "no more", name)
                        self.set_rule(token, "", name)
                        self.set_rule(be, "", name)
                        if any_:
                            self.set_rule(any_, "", name)
                        if subj.lower_ in self.NEGATIVES_NO_MORE:
                            self.set_rule(subj, self.NEGATIVES_NO_MORE[subj.lower_], name)
    
    def wasnt_werent(self, name="wasnt_werent"):
        # feature 163
        for match in re.finditer(r"\bwasn't\b", str(self.doc), re.IGNORECASE):
            self.set_rule_span(match.span(), "weren't", name)
    
    def invariant_tag(self, invariant_tag="isn't it", fronted=False, first_person_subj=False, name="invariant_tag_non_concord"):
        wh = any([token.tag_ in {'WDT', 'WP', 'WP$', 'WRB'} for token in self.tokens])
        for token in self.tokens:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                subj = None
                aux = None
                multiple_aux = False
                sorted_children = sorted(token.children, key=lambda t: t.i)
                for c in sorted_children:
                    if c.dep_ == 'nsubj':
                        subj = c
                    elif c.dep_ == 'aux':
                        if aux:
                            multiple_aux = True
                        else:
                            aux = c
                        
                if subj and ((subj.lower_=='i') or (not first_person_subj)) and (not multiple_aux) and (not wh):
                    predicate = [component for component in self.get_subtree_components_limited(token, []) if component.i > token.i and component!=subj]
                    predicate_text = ' '.join([c.text for c in predicate]) 

                    changed = False
                    if token.lemma_=='be' and subj.idx > token.idx and not aux:
                        # this next check is to avoid manipulating non-question inversions
                        # like "Rarely does one see a Manx cat" or "Not since I was a kid have I felt so scared."
                        if (not len(sorted_children)) or (sorted_children[0].idx >= token.idx):
                            # "Are you...?" | "Is that...?"
                            if fronted:
                                self.set_rule(token, f"{invariant_tag}, {subj} {token.lower_} {predicate_text}", name)
                            else:
                                self.set_rule(token, f"{subj} {token.lower_} {predicate_text}, {invariant_tag}", name)
                            self.set_rule(subj, "", name)
                            changed = True
                    elif aux and aux.lemma_ == 'do' and aux.idx < subj.idx:
                        # this next check is to avoid manipulating non-question inversions
                        # like "Rarely does one see a Manx cat" or "Not since I was a kid have I felt so scared."
                        if (not len(sorted_children)) or (sorted_children[0].idx >= aux.idx):
                            # "Did you...?"
                            if fronted:
                                self.set_rule(token, f"{invariant_tag}, {subj} {token._.inflect(aux.tag_).lower()} {predicate_text}", name)
                            else:
                                self.set_rule(token, f"{subj} {token._.inflect(aux.tag_).lower()} {predicate_text}, {invariant_tag}", name)
                            self.set_rule(subj, "", name)
                            self.set_rule(aux, "", name)
                            changed = True
                    elif aux and aux.idx < subj.idx:
                        # this next check is to avoid manipulating non-question inversions
                        # like "Rarely does one see a Manx cat" or "Not since I was a kid have I felt so scared."
                        if (not len(sorted_children)) or (sorted_children[0].idx >= aux.idx):
                            # Other forms of questions by inverting the auxiliary (non-wh)
                            if fronted:
                                self.set_rule(token, f"{invariant_tag}, {subj} {aux.lower_} {token.lower_} {predicate_text}", name)
                            else:
                                self.set_rule(token, f"{subj} {aux.lower_} {token.lower_} {predicate_text}, {invariant_tag}", name)
                            self.set_rule(subj, "", name)
                            self.set_rule(aux, "", name)
                            changed = True
                    if changed:
                        for component in predicate:
                            self.set_rule(component, "", name)
    
    def invariant_tag_amnt(self):
        # feature 164
        self.invariant_tag(invariant_tag="amn't I", fronted=False, first_person_subj=True, name="invariant_tag_amnt")
    
    def invariant_tag_non_concord(self):
        # feature 165
        self.invariant_tag(invariant_tag="isn't it", fronted=False, first_person_subj=False, name="invariant_tag_non_concord")
        
    def invariant_tag_can_or_not(self):
        # feature 166
        self.invariant_tag(invariant_tag="can or not", fronted=False, first_person_subj=False, name="invariant_tag_can_or_not")
        
    def invariant_tag_fronted_isnt(self):
        # feature 167
        self.invariant_tag(invariant_tag="isn't", fronted=True, first_person_subj=True, name="invariant_tag_fronted_isnt")
                 
    def uninflect(self):
        # feature 170
        """
        change all present tense verbs to 1st person singular form
        """
        for token in self.tokens:
            # we don't want to change contractions or irregulars
            if (token.lower_ not in {"am", "is", "are"}) and (
                not self.is_contraction(token)
            ):
                if token.tag_ in {"VBZ", "VBP"}:  # simple present tense verb
                    uninflected = token._.inflect("VBP", form_num=0)
                    if not uninflected:
                        continue
                    if token.text != str(
                        uninflected
                    ):  # we need to remember this change
                        self.set_rule(token, uninflected, "uninflect")
                        
    def generalized_third_person_s(self):
        # feature 171
        for token in self.tokens:
            # we don't want to change contractions or irregulars
            if (token.lower_ not in {"am", "is", "are"}) and (
                not self.is_contraction(token)
            ):
                if token.tag_ == 'VBP':  # simple present tense verb
                    uninflected = token._.inflect("VBZ")
                    if not uninflected:
                        continue
                    if token.text != str(
                        uninflected
                    ):  # we need to remember this change
                        self.set_rule(token, uninflected, "generalized_third_person_s")
                
    def existential_there_it(self, p=0, name="existential_there_it"):
        replace = "there" if (p or random.random() < p) else "it"
        for token in self.tokens:
            if (token.dep_ == "expl") and (str(token).lower() == "there"):
                be = token.head
                be_replace = 'was' if be.tag_ == 'VBD' else 'is'
                if "'" in be.text:
                    be_replace = ' was' if be.tag == 'VBD' else "'s"
                self.set_rule(token, replace, name)
                self.set_rule(be, be_replace, name)
                
    def existential_there(self):
        # feature 172
        self.existential_there_it(p=1, name="existential_there")
        
    def existential_it(self):
        # feature 173
        self.existential_there_it(p=0, name="existential_it")
    
    def drop_aux(self, gonna_filter=False, consider_wh_questions=False, consider_yn_questions=False, progressive_filter=False, NP_filter=False, AP_filter=False, locative_filter=False, be_filter=False, have_filter=False, name="drop_aux"):        
        
        def gonna(token):
            return (token.lower_ in {"gonna", "gunna"}) or (token.lemma_ in {"go", "gon"})
        
        def wh_question(token):
            return ('?' in {c.lemma_ for c in token.head.children}) or (token.tag_ in {'WDT', 'WP', 'WP$', 'WRB'})
        
        def yn_question(token):
            return ('?' in {c.lemma_ for c in token.head.children}) or (token.lemma_ == 'do')
        
        def progressive(token):
            return token.head.tag_ == 'VBG'
        
        def before_NP(token):
            return any([(c.dep_=='attr') and (c.i>token.i) for c in token.children])
        
        def before_AP(token):
            return any([(c.dep_=='acomp') and (c.tag_=='JJ') and (c.i>token.i) for c in token.children])
        
        def before_locative(token):
            return any([(c.tag_=='IN') and (c.i>token.i) for c in token.children])
        
        for token in self.tokens:
             
            if ((token.lemma_ == "be") or (not be_filter)):
                if ((token.lemma_ == "have") or (not have_filter)):
                    if (wh_question(token) or not consider_wh_questions):
                        if (yn_question(token) or not consider_yn_questions):
                            if not self.is_contraction(token):  # we don't want to change contractions
                                if token.lemma_ == "be":  # copulas are a separate case
                                    if (not gonna_filter or gonna(token.head)) and \
                                    (not progressive_filter or progressive(token)) and \
                                    (not NP_filter or before_NP(token)) and \
                                    (not AP_filter or before_AP(token)) and \
                                    (not locative_filter or before_locative(token)):
                                        if token.lower_ in {"is", "are"}:
                                            # we don't want to mess with relative clauses
                                            # e.g. "are" in "That's just what you are today"
                                            # also don't mess with cases that are modified
                                            if (
                                                not self.is_negated(token)
                                                and ("comp" not in token.dep_)
                                                and (not len(list(token.children)) or not any(
                                                    [c.dep_ in {"ccomp", "expl"} for c in token.children]
                                                ))
                                            ):  # and (token.dep_ != 'ROOT'):
                                                self.set_rule(token, "", name)  # drop copula
                                        else:
                                            pass  # don't change past-tense copula
                                elif ((token.dep_ == "aux") and (token.head.dep_ != "xcomp")):
                                    if (not gonna_filter or gonna(token.head)) and \
                                    (not progressive_filter or progressive(token)) and \
                                    (not NP_filter or before_NP(token)) and \
                                    (not AP_filter or before_AP(token)) and \
                                    (not locative_filter or before_locative(token)):
                                        # next, look at other auxilliaries that are not complements
                                        if str(token) == "will":
                                            self.set_rule(token, "gon", name)  # future tense
                                        elif (token.head.lemma_ != "be") and not (
                                            self.is_negated(token.head) or self.is_modal(token)
                                        ):
                                            self.set_rule(token, "", name)  # drop
        
    def drop_aux_be_progressive(self):
        # feature 174
        self.drop_aux(gonna_filter=False, 
                      consider_wh_questions=False, 
                      consider_yn_questions=False,
                      progressive_filter=True,
                      NP_filter=False, 
                      AP_filter=False,
                      locative_filter=False,
                      be_filter=True, 
                      have_filter=False, 
                      name="drop_aux_be_progressive")
        
    def drop_aux_be_gonna(self):
        # feature 175
        self.drop_aux(gonna_filter=True, 
                      consider_wh_questions=False, 
                      consider_yn_questions=False,
                      progressive_filter=False, 
                      NP_filter=False, 
                      AP_filter=False,
                      locative_filter=False,
                      be_filter=True, 
                      have_filter=False, 
                      name="drop_aux_be_gonna") 
        
    def drop_copula_be_NP(self):
        # feature 176
        self.drop_aux(gonna_filter=False, 
                      consider_wh_questions=False, 
                      consider_yn_questions=False,
                      progressive_filter=False, 
                      NP_filter=True, 
                      AP_filter=False,
                      locative_filter=False,
                      be_filter=True, 
                      have_filter=False, 
                      name="drop_copula_be_NP")
        
    def drop_copula_be_AP(self):
        # feature 177
        self.drop_aux(gonna_filter=False, 
                      consider_wh_questions=False, 
                      consider_yn_questions=False,
                      progressive_filter=False, 
                      NP_filter=False, 
                      AP_filter=True,
                      locative_filter=False,
                      be_filter=True, 
                      have_filter=False, 
                      name="drop_copula_be_AP")
        
    def drop_copula_be_locative(self):
        # feature 178
        self.drop_aux(gonna_filter=False, 
                      consider_wh_questions=False, 
                      consider_yn_questions=False,
                      progressive_filter=False, 
                      NP_filter=False, 
                      AP_filter=False,
                      locative_filter=True,
                      be_filter=True, 
                      have_filter=False, 
                      name="drop_copula_be_locative")
        
    def drop_aux_have(self):
        # feature 179
        self.drop_aux(gonna_filter=False, 
                      consider_wh_questions=False, 
                      consider_yn_questions=False,
                      progressive_filter=False, 
                      NP_filter=False, 
                      AP_filter=False,
                      locative_filter=False,
                      be_filter=False,
                      have_filter=True, 
                      name="drop_aux_have")
        
    def drop_aux_gonna_question_progressive(self):
        # feature 174, 175, 228, 229
        self.drop_aux(gonna_filter=True, question_filter=True, progressive_filter=True)
        
    def were_was(self, name="were_was"):
        # feature 180
        for token in self.tokens:
            if token.lower_ == "were" and not ((token.dep_ == "advcl") or ("if" in {c.lower_ for c in token.children})):
                self.set_rule(token, "was", name)
        
    def relativizer(self, replace={'that', 'which', 'who'}, replace_with=['what', 'as', 'at'], name='relativizer'):
        for token in self.tokens:
            if token.lower_ in replace and token.head.dep_ == 'relcl':
                repl = random.choice(replace_with)
                self.set_rule(token, repl, name)
                
    def who_which(self):
        # feature 186
        self.relativizer(replace={'who'}, replace_with=['which'], name='who_which')
        
    def who_as(self):
        # feature 187
        self.relativizer(replace={'who'}, replace_with=['as'], name='who_as')
        
    def who_at(self):
        # feature 188
        self.relativizer(replace={'who'}, replace_with=['at'], name='who_at')
        
    def relativizer_where(self):
        # feature 189
        self.relativizer(replace={'that', 'which'}, replace_with=['where'], name='relativizer_where')
        
    def who_what(self):
        # feature 190
        self.relativizer(replace_with=['what'], name='who_what')
        
    def relativizer_doubling(self, double={'that', 'which', 'who', 'what'}, double_with=['that', 'which', 'what'], name='relativizer_doubling'):
        # feature 191
        for token in self.tokens:
            if token.lower_ in double and token.head.dep_ == 'relcl':
                double_w = double_with.copy()
                first = random.choice(double_w)
                if token.lower_ in double_w:
                    double_w.remove(token.lower_)
                if first in double_w:
                    double_w.remove(first)
                double = random.choice(double_w)
                self.set_rule(token, f"{first} {double}", name)
                
    def analytic_relativizer(self, replace={"whose"}, replace_with = ["that's", "what's"], use_gender_pronoun=True, name='analytic_relativizer'):
        for token in self.tokens:
            if token.lower_ in replace and (token.head.dep_ == 'relcl' or token.head.head.dep_ == 'relcl'):
                replace_w = replace_with.copy()
                if use_gender_pronoun and token.head.head.head.lower_ in {'man', 'guy', 'dude', 'boy', 'son', 'father', 'grandfather', 'grandpa', 'male'}:
                    replace_w = ['that his', 'what his', 'who his']
                    
                if use_gender_pronoun and token.head.head.head.lower_ in {'woman', 'gal', 'girl', 'daughter', 'mother', 'grandmother', 'grandma', 'female'}:
                    replace_w = ['that her', 'what her', 'who her']
                replace = random.choice(replace_w)
                self.set_rule(token, replace, name)
        
    def analytic_whose_relativizer(self):
        # feature 192
        self.analytic_relativizer(replace={"whose"}, replace_with = ["that's", "what's"], name='analytic_whose_relativizer')
    
    def null_relcl_that(self):
        # don't delete "who" because it can be modified in interesting ways by other features
        self.null_relcl(lemmas={'that'})
        
    def null_relcl(self, lemmas={"that", "who"}):
        # feature 193
        for token in self.tokens:
            # relative clause Wh-determiner / Wh-pronoun
            if (token.head.dep_ in {"relcl"}) and (token.dep_ == 'nsubj'):
                self.set_rule(token, "", "null_relcl")  # drop

    def shadow_pronouns(self, name="shadow_pronouns"):
        # feature 194
        # NOTE: nested relative clauses can lead to strange behaviors in the dependency parse
        # e.g., "What's something Spotty did to try to get his attention?"
        for token in self.tokens:
            if token.dep_ == "relcl" and token.pos_ == "VERB" and token.lemma_ != 'be':
                pronoun = "it"
                if (token.head.tag_ in {"NNS", "NNPS"}) or self.token_is_human(
                    token.head
                ):
                    pronoun = "them"

                valid = True
                
                for c in token.children:
                    if c.tag_ in {'WDT', 'WP'}:
                        if c.dep_ == 'nsubj':
                            valid = False
                    elif 'obj' in c.dep_:
                        valid = False
                    elif c.tag_ == 'TO':
                        valid = False
                        
                    for c_ in c.children:
                        if c_.tag_ == 'TO':
                            valid = False
                        
                if valid:
                    self.set_rule(
                        token, token.text + " " + pronoun, name
                    )
                        
    def one_relativizer(self):
        # feature 195
        for token in self.tokens:
            if token.dep_ == 'nsubj' and token.head.dep_=='ROOT': # the matrix subject
                sc = self.get_subtree_components_limited(token, [])
                subject_components = [component for component in sc if component.tag_ not in {'WP', 'WDT'}]
                replace = ' '.join([c.text for c in subject_components] + ['one'])
                self.set_rule(token, replace, "one_relativizer")
                for component in sc:
                    if component!=token:
                        self.set_rule(component, "", "one_relativizer")
                        
    def correlative_constructions(self):
        # feature 196
        for token in self.tokens:
            if token.dep_ == 'relcl':
                that = 'that one'
                if token.head.dep_ == 'nsubj': 
                    if token.head.tag_ == 'NNS': 
                        that = 'those'
                    last_in_NP_idx = token.head.head.i-1
                    if last_in_NP_idx<len(self.tokens):
                        last_in_NP = list(self.tokens)[last_in_NP_idx]
                        self.set_rule(last_in_NP, f"{last_in_NP.text}, {that}", "correlative_constructions")
                        
    def linking_relcl(self):
        # feature 197
        for token in self.tokens:
            # check that children have 'conj' to prevent false positives in phrases like
            # > "That is but one opinion."
            if token.text=='but' and token.dep_=='cc' and token.head.pos_ == 'VERB':#and 'conj' in {c.dep_ for c in token.children}:
                self.set_rule(token, "which", "linking_relcl")
                        
    def preposition_chopping(self):
        # feature 198
        for token in self.tokens:
            if token.dep_ in {"prep", "prt"} and token.head.dep_ in {"relcl", "xcomp", "ccomp"} and token.i>(len(self.tokens)-3):
                self.set_rule(token, "", "preposition_chopping")
                
    def reduced_relative(self):
        # feature 199
        for token in self.tokens:
            if token.dep_ == "acl":  # clausal modifier of noun
                head = token.head  # invert around this word
                for c in token.children:
                    if "obj" in c.dep_:
                        self.set_rule(c, "", "reduced_relative")
                        self.set_rule(token, "", "reduced_relative")
                        self.set_rule(
                            token.head,
                            f"{self.build_complete_phrase(c, name='reduced_relative')} {token.text} {token.head.text}",
                            "reduced_relative",
                        )
                        return
                    elif c.dep_ == "agent" and len(list(c.children)):
                        self.set_rule(c, "", "reduced_relative")
                        self.set_rule(token, "", "reduced_relative")
                        self.set_rule(
                            token.head,
                            f"{self.build_complete_phrase(list(c.children)[0], name='reduced_relative')} {token.text} {token.head.text}",
                            "reduced_relative",
                        )

    def say_complementizer(self):
        # feature 200
        for token in self.tokens:
            if token.lemma_ in self.SPEAK_VERBS or token.lemma_ in {'hear', 'overhear'}:
                for child in token.children:
                    if child.dep_ == 'ccomp':
                        sorted_c = sorted([c for c in child.children], key=lambda c: c.i)
                        if len(sorted_c):
                            first_complement_token = sorted_c[0]
                            if first_complement_token.dep_ == 'mark':
                                # token is the complementizer, so we can replace it with say
                                self.set_rule(first_complement_token, "say", "say_complementizer")
                            else:
                                # complementizer is implicit, so we can just concatenate along with this left token
                                self.set_rule(first_complement_token, f"say {first_complement_token.text}", "say_complementizer")

    def for_complementizer(self):
        # feature 201
        for token in self.tokens:
            if token.dep_ in {'ccomp','xcomp'}:
                for child in token.children:
                    if child.dep_ == 'mark' or child.tag_=='TO':
                        self.set_rule(child, "for", "for_complementizer")
                        
    def for_to_pupose(self):
        # feature 202
        for token in self.tokens:
            if token.dep_ == 'relcl':
                for child in token.children:
                    if child.tag_=='TO':
                        self.set_rule(child, "for to", "for_to_pupose")
    
    def for_to(self):
        # feature 203
        for token in self.tokens:
            if token.tag_ == 'TO':
                self.set_rule(token, "for to", "for_to")

    def what_comparative(self):
        # feature 204
        for token in self.tokens:
            if token.lemma_=='than' and token.dep_=='mark' and token.head.head.tag_=='JJR':
                self.set_rule(token, "than what", "what_comparative")
                
    def existential_got(self):
        # feature 205
        for token in self.tokens:
            if (token.dep_ == "expl") and (token.lower_ == "there") and (token.head.lemma_=='be'):
                self.set_rule(token, "got", "existential_got")
                self.set_rule(token.head, "", "existential_got")
                
    def existential_you_have(self):
        # feature 206
        for token in self.tokens:
            # this transformation isn't specified when the object is negated
            if (token.dep_ == "expl") and (token.lower_ == "there") and (not self.is_negated(token.head)):
                self.set_rule(token, "you have", "existential_you_have")
                self.set_rule(token.head, "", "existential_you_have")
                
    def existential_transformations(self, p=0.5):
        # feature 172, 173, 205, 206
        for token in self.tokens:
            if (token.dep_ == "expl") and (token.lower_ == "there") and (not self.is_negated(token.head)):
                if random.random() < p:
                    replace = "dey"
                    if random.random() < p:
                        replace = "it"
                    self.set_rule(token, replace, "dey_it")
                else:
                    if random.random() < p:
                        self.set_rule(token, "got", "existential_got")
                        self.set_rule(token.head, " ", "existential_got")
                    else:
                        self.set_rule(token, "you have", "existential_you_have")
                        self.set_rule(token.head, " ", "existential_you_have")   
                        
    def that_infinitival_subclause(self):
        # feature 207
        for token in self.tokens:
            subj = ""
            children_tags = {c.tag_ for c in token.children}
            children_deps = {c.dep_ for c in token.children}
            if (
                token.dep_ in {"ccomp"}
                and token.tag_ == "VB"
                and "TO" in children_tags
                and "nsubj" in children_deps
            ):
                for c in token.children:
                    if c.tag_ == "TO":
                        self.set_rule(c, "", "that_infinitival_subclause")
                    elif c.dep_ == "nsubj":
                        self.set_rule(c, "", "that_infinitival_subclause")
                        subj = self.inflect_subject(self.build_transposable_NP(c, 'that_infinitival_subclause', clear_components=True))
                self.set_rule(
                    token, f"that {subj} should {token}", "that_infinitival_subclause"
                )
                
    def drop_inf_to(self):
        # feature 208
        for token in self.tokens:
            if token.tag_ == "TO":
                self.set_rule(token, "", "drop_inf_to") 
                
    def to_infinitive(self):
        # feature 209
        for token in self.tokens:
            if (
                token.dep_ in {"ccomp"}
                and token.tag_ == "VB"
                and "TO" not in {c.tag_ for c in token.children}
                and "aux" not in {c.dep_ for c in token.children}
            ):
                self.set_rule(token, "to " + token.text, "to_infinitive")
                
    def bare_ccomp(self):
        # feature 210
        for token in self.tokens:
            if token.dep_ == 'xcomp' and token.tag_ == 'VBG':
                self.set_rule(token, token._.inflect("VB"), "bare_ccomp")
                
    def clause_final_though_but(self):
        # feature 211
        for token in self.tokens:
            if token.lower_ == 'though' and self.is_clause_final(token):
                self.set_rule(token, "but", "clause_final_though_but")
                
    def clause_final_really_but(self):
        # feature 212
        for token in self.tokens:
            if token.lower_ == 'really' and self.is_clause_final(token):
                self.set_rule(token, "but", "clause_final_really_but")
                
    def get_root(self, token):
        """Returns the root of the sentence to which @token belongs"""
        t = token
        while t.dep_ != "ROOT":
            t = t.head
        return t
                
    def chaining_main_verbs(self):
        # feature 213
        for token in self.tokens:
            # "If" preceeding both the ROOT and the main verb of the relative clause
            # means we can use the chaining construction and simply remove the mark "IF"
            if token.lemma_=='if' and token.idx < self.get_root(token).idx and token.idx < token.head.idx:
                self.set_rule(token, "", "chaining_main_verbs")
                    
    def corr_conjunction_doubling(self):
        # feature 214
        conj_mapping = {'yet': 'still',
                        'but': 'still',
                        'and': 'so'
                       }
        for token in self.tokens:
            if token.dep_ == 'cc' and token.head.pos_=='VERB' and token.lower_ in conj_mapping:
                for child in token.head.children:
                    if child.dep_ == 'conj' and child.pos_=='VERB':
                        if any(['subj' in c.dep_ for c in child.children]):
                            self.set_rule(token, f"{token.lower_} {conj_mapping[token.lower_]}", 'corr_conjunction_doubling')
            elif token.dep_ == 'advmod' and token.lemma_ == 'still':
                for child in token.head.children:
                    if child.lemma_ in {"though", "although", "despite"}:
                        min_idx = self.subtree_min_idx(token.head, use_doc_index=True, exclude_tags={'IN'})
                        self.set_rule(self.tokens[min_idx], f"yet still {self.tokens[min_idx].text}", 'corr_conjunction_doubling')
                        self.set_rule(token, "", 'corr_conjunction_doubling')
                        
    def subord_conjunction_doubling(self):
        # feature 215
        conj_mapping = {'though': 'but',
                          'although': 'but',
                          'whether': 'still',
                          'despite': 'but',
                          'unless': 'still',
                           'so': 'so'}
        for token in self.tokens:
            if token.tag_ == 'IN' and (token.head.head.i > token.head.i):
                min_idx = self.subtree_min_idx(token.head.head, use_doc_index=True)
                if token.lower_ in conj_mapping:
                    self.set_rule(self.tokens[min_idx], f'{conj_mapping[token.lower_]} {self.tokens[min_idx].text}', 'subord_conjunction_doubling')
                    
    def null_prepositions(self):
        # feature 216
        for token in self.tokens:
            if token.dep_ == 'prep':
                self.set_rule(token, "", 'null_prepositions')
                    
    def adj_for_adv(self):
        for token in self.tokens:
            if token.tag_ == 'RB': # adverb
                adj = self.get_adj_form_of_adv(token.text)
                if adj and adj!=token.text and len(adj)<=len(token.text):
                    # this is still an overgeneralization
                    # a solution for reducing false positives could be to find a complete 
                    # list of so called "flat adverbs" 
                    # (http://en.enlizza.com/flat-adverbs-adverbs-coinciding-with-the-adjectives-in-the-english-language/)
                    # -- see flat_adj_for_adv
                    self.set_rule(token, adj, "adj_for_adv")
                    
    def degree_adj_for_adv(self):
        # feature 220
        for token in self.tokens:
            if token.tag_ == 'RB' and token.head.pos_ in {'ADJ', 'ADV'} and token.lower_ in self.DEGREE_MODIFIER_ADV_ADJ:
                self.set_rule(token, self.DEGREE_MODIFIER_ADV_ADJ[token.lower_], 'degree_adj_for_adv')
                
    def flat_adj_for_adv(self):
        # feature 221
        for token in self.tokens:
            if token.tag_ == 'RB' and token.head.pos_ == 'VERB' and token.lower_ in self.FLAT_ADV_ADJ:
                self.set_rule(token, self.FLAT_ADV_ADJ[token.lower_], 'flat_adj_for_adv')
                    
    def too_sub(self, name="too_sub"):
        # feature 222
        for match in re.finditer(r"\bvery\b", str(self.doc), re.IGNORECASE):
            self.set_rule_span(match.span(), "too", name)
    
    def clefting(self):
        # feature 223
        # it + be + X + subordinate clause
        dobj = []
        iobj = []
        aux = []
        subj = None
        root = None
        predicate = []
        min_idx = min([t.i for t in self.tokens])
        for token in self.tokens:
            if token.dep_ == 'ROOT' and token.pos_=='VERB':
                root = token
                for c in root.children:
                    if c.dep_ == 'dobj':
                        dobj.append(c)
                    elif c.dep_ == 'iobj':
                        iobj.append(c)
                    elif c.dep_ == 'nsubj':
                        subj = c
                    elif c.dep_ == 'aux':
                        aux.append(c)
                predicate = [component for component in self.get_subtree_components(root, []) if component.i > root.i]
                if len(predicate) and predicate[-1].dep_=='punct':
                    predicate = predicate[:-1]
                
        if root and subj and len(dobj)==1 and len(iobj)==0 and (subj not in predicate) and not any([a in predicate for a in aux]):
            subj_components = self.get_subtree_components(subj, [])
            left_subj_token = subj_components[0]
            predicate_text = ' '.join([c.text for c in predicate])
            subj_text = ' '.join([c.lower_ if i==0 else c.text for i, c in enumerate(subj_components)])
            #verb_text = ' '.join([c.text for c in sorted([root] + aux, key=lambda t: t.i)])
            # TO DO: check which complementizer to use: that, who, which, etc.
            complementizer = 'that'
            it = "It" if left_subj_token.i==min_idx else 'it'
            self.set_rule(left_subj_token, f"{it} is {predicate_text} {complementizer} {subj_text}", "clefting")
            for c in predicate + subj_components:
                if c != left_subj_token:
                    self.set_rule(c, "", "clefting")
            
    def fronting_pobj(self, p=1):
        # feature 224
        pobj_tokens = []
        for token in self.tokens:
            if token.dep_ == 'pobj' and token.head.dep_ == 'prep':
                pobj_tokens.extend(self.get_subtree_components(token, []))
                pobj_tokens.append(token.head)
        
        if len(pobj_tokens) and random.random() < p:
            pobj_tokens = sorted(list(set(pobj_tokens)), key=lambda t: t.i)
            clause_origin = self.get_clause_origin(token)
            if clause_origin.i < pobj_tokens[0].i:
                pobjs = ' '.join([c.text for c in pobj_tokens])
                self.set_rule(clause_origin, f"{pobjs} {clause_origin.text}", "fronting_pobj")
                for p in pobj_tokens:
                    self.set_rule(p, "", "fronting_pobj")
                
    
    def negative_inversion(self, name="negative_inversion"):
        # feature 226
        for token in self.tokens:
            consider = None
            if token.lower_ == "no":
                consider = token.head
            elif str(token) in self.NEGATIVES.values():
                consider = token
            else:
                continue

            if not self.is_clause_initial(consider):
                continue

            found_aux = False
            for child in consider.head.children:
                
                if (child.dep_ == "aux") and (child.lower_ != "to"):
                    found_aux = True
                    if not self.is_contraction(child):
                        replace = "'t" if str(child)[-1]=='n' else "n't"
                        self.set_rule(child, "", name)
                        self.set_rule(
                            token,
                            f"{child.text}{replace} {token.lower_}",
                            name,
                        )

            if not found_aux and token.head.pos_ in "VERB":
                verb = token.head
                do = "do"
                if verb.tag_ == "VBD":
                    do = "did"
                self.set_rule(token, f"{do}n't {token.lower_}", name)
                self.set_rule(verb, verb._.inflect("VB"), name)
                
    def inverted_indirect_question(self):
        # feature 227
        for token in self.tokens:
            # indirect question
            if token.tag_ in {'WP', 'WDT', 'WP$', 'WRB'} and token.head.dep_ in {'ccomp', 'xcomp'}:
                
                VERB = token.head
                if 'TO' in {c.tag_ for c in token.head.children} and token.head.head.pos_ == 'VERB' and token.head.head.i > token.i:
                    VERB = token.head.head
                
                if VERB.head.lemma_ == 'be':
                    # this isn't an indirect question
                    continue
                    
                if 'if' in {c.lower_ for c in VERB.children}:
                    # this isn't an indirect question
                    continue
                
                aux = []
                subj = None
                for c in VERB.children:
                    if c.dep_ == 'aux':
                        aux.append(c)
                    elif c.dep_ == 'nsubj':
                        subj = c
                if subj:
                    subj_components = self.get_subtree_components(subj, [])
                    left_subj_token = subj_components[0]
                    if len(aux) == 1 and aux[0].i > subj.i:
                        aux = aux[0]
                        self.set_rule(left_subj_token, f"{aux.text} {left_subj_token.text}", "inverted_indirect_question")
                        self.set_rule(aux, "", "inverted_indirect_question")
                    elif not len(aux):
                        do = 'did'
                        if VERB.tag_ in {'VB', 'VBP', 'VBZ'}:
                            do = 'does'
                        self.set_rule(left_subj_token, f"{do} {left_subj_token.text}", "inverted_indirect_question")
                        self.set_rule(VERB, VERB._.inflect('VB'), "inverted_indirect_question")
                        
    def drop_aux_wh(self):
        # feature 228
        self.drop_aux(gonna_filter=False, 
                      consider_wh_questions=True, 
                      consider_yn_questions=False,
                      progressive_filter=False, 
                      NP_filter=False, 
                      AP_filter=False,
                      locative_filter=False,
                      be_filter=False,
                      have_filter=False, 
                      name="drop_aux_wh")
        
    def drop_aux_yn(self):
        # feature 229
        self.drop_aux(gonna_filter=False, 
                      consider_wh_questions=False, 
                      consider_yn_questions=True,
                      progressive_filter=False, 
                      NP_filter=False, 
                      AP_filter=False,
                      locative_filter=False,
                      be_filter=False,
                      have_filter=False, 
                      name="drop_aux_yn")
        
    def doubly_filled_comp(self):
        # feature 230
        WPS = []
        WPS_text = set()
        for token in self.tokens:
            if token.tag_ == 'WP' and token.text not in WPS_text:
                WPS.append(token)
                WPS_text.add(token.text)
        if len(WPS) > 1:
            WPS = sorted(WPS, key=lambda t: -t.i)
            replace_wh = " ".join([WP.lower_ for WP in WPS])
            self.set_rule(WPS[-1], replace_wh, 'doubly_filled_comp')
            for WP in WPS[:-1]:
                self.set_rule(WP, "", 'doubly_filled_comp')
                
    def superlative_before_matrix_head(self):
        # feature 231
        for token in self.tokens:
            if token.lemma_ == 'most' and token.dep_ == 'advmod' and token.head.dep_ in {'relcl', 'nsubj'}:
                hh = token.head.head
                if hh.dep_ == 'nsubj':
                    self.set_rule(hh, f"most {hh.text}", 'superlative_before_matrix_head')
                    self.set_rule(token, "", 'superlative_before_matrix_head')
                elif hh.dep_ == 'ROOT':
                    for c in hh.children:
                        if c.dep_ == 'nsubj':
                            self.set_rule(c, f"most {c.text}", 'superlative_before_matrix_head')
                            self.set_rule(token, "", 'superlative_before_matrix_head')
                            
    def double_obj_order(self):
        # feature 232
        objects = []
        for token in self.tokens:
            if 'obj' in token.dep_ or token.head.dep_ == 'dative':
                objects.append(token)
        if len(objects)==2:
            objects = sorted(objects, key=lambda t: -t.i)
            replace_obj = " ".join([obj.text for obj in objects])
            self.set_rule(objects[0], replace_obj, "double_obj_order")
            self.set_rule(objects[-1], "", "double_obj_order")
            if objects[0].dep_ == 'pobj':
                self.set_rule(objects[0].head, "", "double_obj_order")
                
    def acomp_focusing_like(self):
        # feature 234
        for token in self.tokens:
            if token.dep_ == 'acomp':
                cands = sorted([c for c in token.children if c.dep_ == 'advmod']+[token], key=lambda t: t.i)
                self.set_rule(cands[0], f"like {cands[0].text}", "acomp_focusing_like")
        
    def quotative_like(self):
        # feature 235
        for token in self.tokens:
            if token.lemma_ in self.SPEAK_VERBS and token.tag_ == 'VBD':
                num = self.get_verb_number(token)
                was = "was" if num < 2 else "were"
                quote = None
                if token.i+1 < len(self.doc) and self.doc[token.i+1].text in {"'", '"'}:
                    quote = self.doc[token.i+1]
                elif token.i+2 < len(self.doc) and self.doc[token.i+2].text in {"'", '"'}:
                    quote = self.doc[token.i+2]
                if quote: # quotations
                    self.set_rule(
                        token, f"{was} like", "quotative_like"
                    )

    ## Extra features not in eWAVE
                            
    def got(self):
        """Changes verbal present-tense 'have' to 'got' (AAVE)"""
        for token in self.tokens:
            if (
                token.lower_ in {"have", "has"}
                and token.pos_ == "VERB"
                and self.has_object(token)
                and not any([c.dep_ == "aux" for c in token.children])
            ):
                self.set_rule(token, "got", "got")
                            
    def ass_pronoun(self, p=0.1):
        """Implements ass camouflage constructions, reflexive constructions, and intensifiers"""
        for token in self.tokens:
            if (token.dep_ in self.OBJECTS) and (
                token.tag_ == "PRP"
            ):  # the token is an object pronoun
                if (
                    str(token) in self.POSSESSIVES
                ):  # the token has a possessive counterpart
                    self.set_rule(token, self.POSSESSIVES[str(token)] + " ass", "ass")
            elif (
                (token.dep_ == "amod")
                and (token.tag_ == "JJ")
                and self.is_gradable_adjective(str(token))
            ):  # the token is a gradable adjective modifier
                if random.random() < p:
                    self.set_rule(token, str(token) + "-ass", "ass")
            elif token.pos_ == "VERB":
                # TODO: handle imperative: "Get inside" => "Get your ass inside"
                pass
            
    ## Surface features and fixes and lexical substitutions

    def surface_dey_conj(self, string):
        """Fix errors in conjucation left by it/dey construction, etc. (AAVE)"""
        string = re.sub(r"\bDey are", "Dey is", string)
        string = re.sub(r"\bAre dey\b", "Is dey", string)
        string = re.sub(r"\bIt are", "It is", string)
        string = re.sub(r"\bAre it", "Is it", string)
        string = re.sub(r"\bdey are", "dey is", string)
        string = re.sub(r"\bare dey\b", "is dey", string)
        string = re.sub(r"\bit are", "it is", string)
        string = re.sub(r"\bare it", "is it", string)
        return string

    def surface_lexical_sub(self, string, p=0.4):
        """Make all lexical substitutions indicated in the dictionary @self.lexical_swaps"""
        for sae in self.lexical_swaps.keys():
            if ("." not in sae) and len(self.lexical_swaps[sae]):
                for match in re.finditer(r"\b%s\b" % sae, string, re.IGNORECASE):
                    if random.random() < p:  # swap
                        start, end = match.span()

                        swap = random.sample(list(self.lexical_swaps[sae]), 1)[0]
                        string = "%s%s%s" % (string[:start], swap, string[end:])

                        self.modification_counter["surface_lexical_sub"] += 1
                        self.modification_counter["total"] += 1
        return string
    
    def surface_aint_sub(self, string):
        """Substitute ain't in the place of other negations"""
        
        orig = string  # .copy()
        string = re.sub(r"\bgon not\b", "ain't gon", string, flags=re.IGNORECASE)
        string = re.sub(r"\bis no\b", "ain't no", string, flags=re.IGNORECASE)
        string = re.sub(r"\bare no\b", "ain't no", string, flags=re.IGNORECASE)
        string = re.sub(r"\bare no\b", "ain't no", string, flags=re.IGNORECASE)
        string = re.sub(r"\bis not\b", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bisn['‘’`]?t", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bare not\b", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\baren['‘’`]?t", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bhaven['‘’`]?t", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bhasn['‘’`]?t", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bhadn['‘’`]?t", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bdidn['‘’`]?t", "ain't", string, flags=re.IGNORECASE)

        if string != orig:
            self.modification_counter["surface_aint_sub"] += 1
            self.modification_counter["total"] += 1
            
        return string
    
    def surface_fix_contracted_copula(self, string):
        # added to fix "ain't" being squished onto the preceding token due to contracted copula
        string = re.sub(r"([^\s])ain't", r"\1 ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"([^\s])isn't", r"\1 isn't", string, flags=re.IGNORECASE)
        return string
    
    def surface_fix_spacing(self, string):
        # added to fix "ain't" being squished onto the preceding token due to contracted copula
        string = re.sub(r" \.$", ".", string, flags=re.IGNORECASE)
        string = re.sub(r" \?$", "?", string, flags=re.IGNORECASE)
        string = re.sub(r" !$", "!", string, flags=re.IGNORECASE)
        string = re.sub(r" $", "", string, flags=re.IGNORECASE)
        string = re.sub(r"^ ", "", string, flags=re.IGNORECASE)
        return string
    
    def surface_fixes(self, string):
        string = self.surface_fix_contracted_copula(string)
        string = self.surface_fix_spacing(string)
        string = re.sub(r"nn't", "n't", string, flags=re.IGNORECASE)
    
    ## Helper functions
                
    def is_vowel(self, letter):
        return letter in "AaEeIiOoUu"
                
    #https://speakspeak.com/resources/english-grammar-rules/english-spelling-rules/double-consonant-ed-ing
    def double_consonant(self, verb):
        """Returns true if @verb should bear a double consonant before adding the past-tense -ed morpheme"""
        vowels = "AaEeIiOoUu"

        syllables = self.get_syllable_stress(verb)
        if len(verb)<3 or not len(syllables):
            # Verb is too short to analyze patterns, or it is not contained in cmudict
            return False
        elif verb[-1] in {'w', 'y'}:
            # We don't double the 'w' or the 'y' sound
            return False
        elif (not self.is_vowel(verb[-1])) and (not self.is_vowel(verb[-2])):
            # Don't double the consonant when the final letter has two consonants
            return False
        elif (self.is_vowel(verb[-2])) and (self.is_vowel(verb[-3])):
            # Don't double the consonant when two vowels come directly before it
            return False                   
        elif len(syllables)==1:
            # We double the final consonant when a mono-syllabic word ends in consonant + vowel + consonant.
            return (not self.is_vowel(verb[-3])) and (self.is_vowel(verb[-2])) and (not self.is_vowel(verb[-1]))
        else:
            # We double the final consonant when a multi-syllabic word has its final syllable stressed
            return syllables[-1]>syllables[-2]

    def get_syllable_stress(self, word):
        """Returns a list representing the syllabic stress 
        (e.g. 'happen' --> [1, 0]; 'begin' --> [0, 1])"""

        if word not in self.cmudict:
            return []
        
        syllables_found = []
        for phone in self.cmudict[word][0]:
            bare = re.split('[0-9]+', phone)
            number = re.findall('[0-9]+', phone)
            if len(number):
                syllables_found.append(int(number[0]))
        return syllables_found

    def subj_sentence_agreement(self, token, name, form_num=0):
        for c in list(token.head.children) + [token.head]:
            if c.tag_ in {"VBP", "VBZ", "VBD"}:
                new_tag = "VBP" if c.tag_ == "VBZ" else c.tag_
                self.set_rule(c, c._.inflect(new_tag, form_num=form_num), name)
        for c in token.children:
            if c.tag_ == "DT" and c.text in self.PLURAL_DETERMINERS.keys():
                self.set_rule(c, self.PLURAL_DETERMINERS[c.text], name)
                    
    def has_additional_aux(self, verb):
        """Returns TRUE if @verb has an auxiliary other than 'have'/'has'"""
        for child in verb.children:
            if child.dep_ in {'aux', 'auxpass'} and child.lemma_ != 'have':
                return True
        return False

    def is_indefinite_noun(self, token):
        """returns boolean value indicating whether @token is an indefinite noun"""
        # A noun may indefinite if either it has no determiner or it has an indefinite article
        # It cannot be a proper noun, pronoun, possessive, or modifier, nor can it be negated
        return (
            (token.pos_ in {"NOUN"})
            and (token.tag_ not in {"PRP", "NNP"})
            and (str(token) not in self.NEGATIVES.values())
            and not any(
                [c.dep_ in {"poss", "amod", "neg", "advmod"} for c in token.children]
            )
            and not any(
                [
                    (c.dep_ == {"det", "nummod"}) and (c.lemma_ not in {"a", "an"})
                    for c in token.children
                ]
            )
        )
    
    def get_children_deps(self, token):
        return {c.dep_ for c in token.children}

    def get_clause_origin(self, token):
        """Returns the root or parent node in the clause to which @token belongs"""
        t = token
        while ("comp" not in t.dep_) and (t.dep_ not in {"ROOT", "conj"}):
            t = t.head
        return t
        
    def subtree_min_idx(self, token, use_doc_index=False, use_max_idx=False, exclude_tags={}):
        """Returns the minimum character index or the left side of the clause to which @token belongs"""
        indices = []
        for child in token.children:
            if ("comp" not in child.dep_) and child.dep_ not in {"punct", 'advcl'} and child.tag_ not in exclude_tags:
                indices.append(self.subtree_min_idx(child, use_doc_index=use_doc_index, use_max_idx=use_max_idx))
        if use_doc_index:
            indices.append(token.i)
        else:
            indices.append(token.idx)
        if use_max_idx:
            return max(indices)
        return min(indices)
    
    def subtree_max_idx(self, token, use_doc_index=False, use_max_idx=False, exclude_tags={}):
        """Returns the maximum character index or the right side of the clause to which @token belongs"""
        return self.subtree_min_idx(token, use_doc_index, use_max_idx=True, exclude_tags=exclude_tags)

    def is_clause_initial(self, token):
        """Returns boolean value indicating whether @token is the first token in its clause (useful for negative inversion)"""
        root = self.get_clause_origin(token)
        return token.idx == self.subtree_min_idx(root)
    
    def is_clause_final(self, token):
        """Returns boolean value indicating whether @token is the first token in its clause (useful for negative inversion)"""
        root = self.get_clause_origin(token)
        return token.idx == self.subtree_max_idx(root)
                
    def noun_is_modified(self, noun):
        return any(['mod' in c.dep_ for c in noun.children])
                    
    def is_singular_noun(self, noun):
        infl = self.inflection.singular_noun(noun)
        if not infl:
            return self.wnl.lemmatize(noun, "n") == noun
        return self.inflection.singular_noun(noun) == noun

    def get_verb_number(self, verb):
        for c in verb.children:
            if c.dep_ == "nsubj":
                if c.lower_ == 'i':
                    return 0
                if c.lower_ in {'he', 'she', 'it', 'who', 'what'}:
                    return 1
                elif (c.lower_ != 'you') and self.is_singular_noun(c.text):
                    return 1
                else:
                    return 2
        return 2           
                        
    def has_form(self, word, form='v'):
        for synset in wn.synsets(word):
            if synset.pos() == form:
                return True
        return False

    def token_is_human(self, token):
        return (token.lower_ in self.HUMAN_NOUNS) and (
            token.pos_ in ["NOUN", "PRON", "PROPN"]
        )
                
    def get_subj(self, token):
        subj = None
        for child in token.children:
            if 'subj' in child.dep_:
                subj = child
        return subj
    
    def get_subj_matrix(self, token):
        """Tries to retrieve the subject of the verb @token, and if the subject is not found, the function searches in the matrix clause, then the matrix around that clause and so on recursively until a subject is found or the ROOT of the parse is reached"""
        current = token
        subj = self.get_subj(token)
        while (not subj) and (current.dep_ != 'ROOT'):
            current = current.head
            subj = self.get_subj(current)
        return subj
        
    def inflect_subject(self, text):
        mapping = self.PRONOUN_OBJ_TO_SUBJ
        if text in mapping:
            return mapping[text]
        return text
    
    def get_subtree_components(self, token, components=[]):
        components.append(token)
        for child in token.children:
            self.get_subtree_components(child, components)
        return sorted(components, key=lambda t: t.i)
    
    def get_subtree_components_limited(self, token, components=[], exclude_deps={'conj', 'cc', 'punct'}):
        components.append(token)
        for child in token.children:
            if child.dep_ not in exclude_deps:
                self.get_subtree_components_limited(child, components)
        return sorted(components, key=lambda t: t.i)
    
    def build_transposable_NP(self, token, name, clear_components=False):
        """Return the string form of @token as a full NP with all of its dependants"""
        # before, this used to also clear out the dependents in the rule set so that the @token NP is "transposable" or movable throughout the rule set
        components = self.get_subtree_components(token, [])
        if clear_components:
            for c in components:
                self.set_rule(c, "", name)
        components = [t.text for t in sorted(components, key=lambda t: t.i)]
        return " ".join(components)

    def build_complete_phrase(self, head, erase_trace=True, name=""):
        candidates = [head] + list(head.children)
        if erase_trace:
            for cand in candidates:
                self.set_rule(cand, "", name)

        candidates = {(candidate, candidate.idx) for candidate in candidates}
        return "-".join(
            [cand[0].text for cand in sorted(candidates, key=lambda x: x[1])]
        )
                        
    def has_clause_subordination_marker(self, verb):
        return any([c.dep_=='mark' for c in verb.children])
               
    def has_concrete_object(self, token, concreteness_threshold=4.0):
        """returns boolean value indicating whether @token has an object dependant (child) that is sufficiently concrete"""
        return any([ (c.dep_ in self.OBJECTS) and (c.lemma_ in self.NOUN_CONCRETENESS) and (self.NOUN_CONCRETENESS[c.lemma_]>=concreteness_threshold) for c in token.children])
                
    def has_prt_child(self, token):
        return any([c.dep_=='prt' for c in token.children])
           
    def clear_subtree(self, root, name=""):
        for child in root.children:
            self.set_rule(child, "", name)
            self.clear_subtree(child, name=name)         
                
    def get_adj_form_of_adv(self, adv):
        possible_adj = []
        for ss in wn.synsets(adv):
            for lemmas in ss.lemmas(): 
                for ps in lemmas.pertainyms():
                    possible_adj.append(ps.name())

        edit_distances = np.array([edit_distance(adv, adj) for adj in possible_adj])
        if not len(edit_distances):
            return None
        return possible_adj[np.argmin(edit_distances)]

    def remove_recursive(self, token):
        """Remove @token and all children of @token in the dependency tree"""
        if not token:
            return
        self.set_rule(token, "")
        for child in token.children:
            self.remove_recursive(child)
            
    def rules_to_idx_vec(self, rules, vec_len):
        vec = np.zeros(vec_len)
        for key in rules.keys():
            start, stop = key
            for i in range(start, stop+1):
                vec[i] = 1
        return vec

    def decide_executable_rules(self, rules, covered_len):
        # resolves conflicts between rules by iteratively sampling from 
        # distribution in self.morphosyntax_transforms and discarding 
        # any rules whose target span overlaps with those already chosen
        
        executable = []
        options = list(rules.keys())
        weights = [self.morphosyntax_transforms[f] if f in self.morphosyntax_transforms else 1.0 for f in options]

        covered = np.zeros(covered_len)

        while len(options):
            rule_name = random.choices(options, weights=weights, k=1)[0]
            rule = rules[rule_name]
            vec = self.rules_to_idx_vec(rule, covered_len)
            idx = options.index(rule_name)

            if not any(vec*covered):
                for rule_span in self.rules[rule_name].keys():
                    executable.append((rule_span, self.rules[rule_name][rule_span]))
                covered += vec
            else:
                # clash! throw it out
                pass

            options.pop(idx)
            weights.pop(idx)

        return dict(executable)
    
    def decide_filter_rules(self):
        # consider the attestation probabilities and use these to decide which rules are expressed 
        # in the surface realization.
        decided_rules = {}
        for rule_name in self.rules.keys():
            feature_attestation = 1.0
            if rule_name in self.morphosyntax_transforms:
                feature_attestation = self.morphosyntax_transforms[rule_name]
            if random.random() < feature_attestation:
                decided_rules[rule_name] = self.rules[rule_name].copy()
        return decided_rules

    def compile_from_rules(self):
        """
        compile all accumulated morphosyntactic rules

        rules will be a dictionary of the following key/value form:

        """
        executable_rules = self.decide_executable_rules(rules=self.decide_filter_rules(), 
                                                        covered_len=len(str(self.doc)))
        
        prev_idx = 0
        compiled = ""
        for indices in sorted(executable_rules.keys()):
            start_idx, end_idx = indices
            compiled += self.string[prev_idx:start_idx]
            prev_idx = end_idx + 1

            compiled += executable_rules[indices]["value"]

        if prev_idx < len(self.string):
            compiled += self.string[prev_idx:]
            
        self.executed_rules = executable_rules
        return compiled

    def highlight_modifications_html(self):
        """Return an HTML highlighting and indexing of all modified tokens (used for MTurk validation)"""
        prev_idx = 0
        compiled = ""
        j = 1
        for indices in sorted(self.executed_rules.keys()):
            start_idx, end_idx = indices
            compiled += self.string[prev_idx:start_idx]
            prev_idx = end_idx + 1

            compiled += (
                ("<a href='%s' title='%s'><mark>" % (self.executed_rules[indices]["type"], j))
                + self.string[start_idx:end_idx+1]
                + "</mark></a>"
            )
            j += 1

        if prev_idx < len(self.string):
            compiled += self.string[prev_idx:]
        return compiled