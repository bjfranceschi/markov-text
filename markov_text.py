"""
Use like:
from markov_text import MarkovText
m = MarkovText()
m.train('markov_text_2.txt')
m.generate()
"""

import re
import random
from multiprocessing.pool import ThreadPool

import pandas as pd

class MarkovText:
    def __init__(self):
        pass

    def _pair_tokens(self, i):
        ngram = tuple(self.tokens[i:i+self.num_grams])
        token = self.tokens[i+self.num_grams]

        if self.tokens[i-1] in self.end_puncts:
            start_gram = True
        else:
            start_gram = False

        return {'ngram': ngram, 'next_token': token, 'starting_ngram?': start_gram}

    def train(self, filename, num_grams=2, tokens=None):
        self.filename = filename
        self.num_grams = num_grams
        self.tokens = tokens
        self.end_puncts = ['.', '!', '?']

        reference_file = open(filename, 'r')
        string_text = reference_file.read()
        string_text = string_text.lower()
        string_text = re.sub(r'\n', ' ', string_text)
        string_text = re.sub(r'“', '"', string_text)
        string_text = re.sub(r'”', '"', string_text)

        # add spaces to certain chars so they're considered as tokens when split
        self.punct_chars = [',', '\.', '!', '\?', ':', '—', '-', '"']
        for char in self.punct_chars:
            string_text = re.sub(char, " "+re.sub(r'\\', '', char), string_text)

        # build list of all tokens, filtering out '' (originally line breaks)
        self.tokens = string_text.split(' ')
        self.tokens = [i for i in self.tokens if i != '']

        # add a token denoting end of text because otherwise the last ngram won't have a 
        # corresponding "next token", let alone one which lets the model know the end comes next
        self.tokens.append('|*END*OF*TEXT*|')

        ### build "trained" df
        model = pd.DataFrame(columns=['ngram', 'next_token'])

        with ThreadPool(250) as p:
            paired_tokens = p.map(self._pair_tokens, range(0,len(self.tokens)-num_grams))

        model = pd.DataFrame(paired_tokens)

        ngram_counts = pd.DataFrame(model['ngram'].value_counts()).reset_index().rename(columns={'index': 'ngram', 'ngram': 'ngram_count'})
        token_counts = model.groupby(['ngram', 'next_token']).size().reset_index(name="next_token_for_ngram_count")

        model = pd.merge(model, ngram_counts, on='ngram')
        model = pd.merge(model, token_counts, on=['ngram', 'next_token'])
        model.drop_duplicates(subset=['ngram', 'next_token'], inplace=True)
        model['next_token_probability'] = model['next_token_for_ngram_count']/model['ngram_count']
        
        self.model = model.reset_index(drop=True)

    def _return_token(self, gram):
        subset = self.model[self.model['ngram'] == gram]
        #print("SUBSET: ", subset)
        generated_word = random.choices(list(subset['next_token']), weights=list(subset['next_token_probability']))
        return generated_word

    def _cleanup_generated_text(self):

        # remove previously added spaces from certain chars ("this , and" -> "this, and")
        for char in self.punct_chars:
            self.generated_text = re.sub(f" {char}", re.sub(r'\\', '', char), self.generated_text)
        
        # if only one quote, add closing quote before the following end-punctuation mark
        # also should do same for parentheses...
        # also rather than starting totally randomly, should start randomly from
        # actual beginnings of sentences
        if self.generated_text.count('"') == 1:

            end_punct_positions = [pos for pos in sorted([self.generated_text.rfind('.'), self.generated_text.rfind('!'), self.generated_text.rfind('?')]) if pos != -1]
            
            # if quote is a beginning quote
            if re.search('"[a-zA-Z|0-9]', self.generated_text):
                for i in end_punct_positions:
                    if i > self.generated_text.find('"'):
                        insert_position = i
                        break

                # if insert_position var doesn't exist, means no end punctuation - stick " on end
                if 'insert_position' not in locals():
                    insert_position = len(self.generated_text)
            
            # if quote is a closing quote
            elif re.search('\S"', self.generated_text):
                for i in end_punct_positions:
                    if i < self.generated_text.find('"'):
                        insert_position = i
                        break

                # if insert_position var doesn't exist, means no end punctuation - stick " on beginning
                if 'insert_position' not in locals():
                    insert_position = 0
            else:
                print("else??")
                print(self.generated_text)

            self.generated_text = f'{self.generated_text[:insert_position]}"{self.generated_text[insert_position:]}'

        self.generated_text = self.generated_text.capitalize()

    def generate(self, max_tokens=random.randrange(10, 40)):
        self.max_tokens = max_tokens

        # start with a "starting ngram" - i.e. one that begins a sentence
        generated_text = []
        seed = random.choice(list(self.model[self.model['starting_ngram?'] == True]['ngram']))
        generated_text.extend([token for token in seed])

        # generate text
        for i in range(max_tokens):
            gram = tuple(generated_text[-2:])
            #print("GRAM: ", gram)
            
            # generate text as long as there's no end token
            if '|*END*OF*TEXT*|' not in gram:
                generated_text.extend(self._return_token(gram))

        # truncate at last stop-punctuation-mark
        for i in range(1, len(generated_text)):
            if generated_text[-i] in self.end_puncts:
                generated_text = generated_text[:-i+1]
                break

        self.generated_text = ' '.join(generated_text)

        #clean up text
        self._cleanup_generated_text()

        return self.generated_text





