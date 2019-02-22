# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:27:34 2019

@author: Anshuman_Mahapatra
"""

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
##Display in Web
from spacy import displacy
text = """But Google is starting from behind. The company made a late push
into hardware, and Apple’s Siri, available on iPhones, and Amazon’s Alexa
software, which runs on its Echo and Dot devices, have clear leads in
consumer adoption."""



nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
displacy.serve(doc, style='ent')


##visible in http://localhost:5000/

doc = """Apple is looking at buying U.K. startup for $1 billion"""
doc1 = nlp(doc)
displacy.serve(doc1, style='ent')


######few examples
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(u'San Francisco considers banning sidewalk delivery robots')

# document level
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print(ents)

# token level
ent_san = [doc[0].text, doc[0].ent_iob_, doc[0].ent_type_]
ent_francisco = [doc[1].text, doc[1].ent_iob_, doc[1].ent_type_]
print(ent_san)  # [u'San', u'B', u'GPE']
print(ent_francisco)  # [u'Francisco', u'I', u'GPE']

#the position referred is from start and end in the doc (here even though the token is ending in 12 we are specifying 13)



##

from spacy.tokens import Span

nlp = spacy.load('en_core_web_sm')
doc = nlp(u"FB is hiring a new Vice President of global policy")
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print('Before', ents)


# the model didn't recognise "FB" as an entity :(

##So modify and update it 
ORG = doc.vocab.strings[u'ORG']  # get hash value of entity label
fb_ent = Span(doc, 0, 1, label=ORG) # create a Span for the new entity
doc.ents = list(doc.ents) + [fb_ent]

ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print('After', ents)
# [(u'FB', 0, 2, 'ORG')] 🎉


###more examples
import numpy
import spacy
from spacy.attrs import ENT_IOB, ENT_TYPE

nlp = spacy.load('en_core_web_sm')
doc = nlp.make_doc(u'London is a big city in the United Kingdom.')
print('Before', list(doc.ents))  # []

header = [ENT_IOB, ENT_TYPE]
attr_array = numpy.zeros((len(doc), len(header)))
attr_array[0, 0] = 3  # B
attr_array[0, 1] = doc.vocab.strings[u'GPE']
doc.from_array(header, attr_array)
print('After', list(doc.ents))  # [London]

###Training own Data

import os
import random
from spacy.gold import offsets_from_biluo_tags
os.chdir("D:\Data Science\POC\spacy")   

train_data = [
    ("Uber blew through $1 million a week", [(0, 4, 'ORG')]),
    ("Android Pay expands to Canada", [(0, 11, 'PRODUCT'), (23, 30, 'GPE')]),
    ("Spotify steps up Asia expansion", [(0, 8, "ORG"), (17, 21, "LOC")]),
    ("Google Maps launches location sharing", [(0, 11, "PRODUCT")]),
    ("Google rebrands its business apps", [(0, 6, "ORG")]),
    ("look what i found on google! 😂", [(21, 27, "PRODUCT")])]


optimizer = nlp.begin_training()
for itn in range(100):
    random.shuffle(train_data)
    for raw_text, entity_offsets in train_data:
        doc = nlp.make_doc(raw_text)
        gold = GoldParse(doc, entities=entity_offsets)
        nlp.update([doc], [gold], drop=0.5, sgd=optimizer)
nlp.to_disk('/model')