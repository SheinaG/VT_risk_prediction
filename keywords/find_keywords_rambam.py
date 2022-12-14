import pandas as pd
from deep_translator import GoogleTranslator
from fuzzywuzzy import fuzz, process
from rbaf_parser import *

debug = 0

NEGATE = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite", "no"]

path_to_rbdf = '/MLAIM/AIMLab/Shany/databases/rbafdb/'
path_to_dictionary = path_to_rbdf + 'documentation/reports/RBAF_reports.xlsx'
path_to_keywords = path_to_rbdf + 'documentation/reports/Annotation_Codes.xlsx'

my_dictionary = pd.read_excel(path_to_dictionary, engine='openpyxl')
keywords = pd.read_excel(path_to_keywords, engine='openpyxl')

holter_pd = pd.DataFrame(columns=['AFIB', 'VT'])
names = keywords['Code']
keywords = keywords.transpose()

if debug:
    debug_ids = ['7A17D780', '2620Da2f', '2418C117']
    # debug_ids = np.load('/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/reports/control_group.npy')

    my_dictionary['holter_id'] = my_dictionary['holter_id'].astype(str)
    my_dictionary = my_dictionary.loc[my_dictionary['holter_id'].isin(debug_ids)]

for i, id in enumerate(my_dictionary['holter_id']):
    j = int(my_dictionary['holter_id'][my_dictionary['holter_id'] == id].index.values)
    text_a = my_dictionary['טקסט מתוך סיכום'][j]
    text_b = my_dictionary['טקסט מתוך תוצאות'][j]
    if pd.isna(text_a) or text_a == 0 or text_a == ' ':
        text = ''
    else:
        text = text_a
    if not (pd.isna(text_b) or text_b == 0 or text_b == ' '):
        text = text + text_b
    if text == '':
        continue
    big_lines = text.splitlines()
    list_lines = []
    list_lines_2 = []
    for b_l in big_lines:
        list_lines.append(b_l.split('.'))
    lines = [item for sublist in list_lines for item in sublist]
    # for s_l in small_lines:
    #     list_lines_2.append(s_l.split(','))
    # lines = [item for sublist in list_lines_2 for item in sublist]
    # for AFIB:
    words_list = list(set(keywords[1].dropna()))
    words_list.append('AFIB')
    result = 0
    for line in lines:
        if line == ' ':
            continue
        for word in words_list:
            highest = fuzz.partial_ratio(line, word)
            if highest >= result:
                result = highest
                result_word = word
                best_line = line

    if result > 80:
        translated_line = GoogleTranslator(source='auto', target='en').translate(best_line)
        translated_word = GoogleTranslator(source='auto', target='en').translate(result_word)
        negativ = process.extractOne(translated_line, NEGATE, scorer=fuzz.token_set_ratio)
        if negativ[1] > 40:
            translated_line_list = translated_line.split(' ')
            closest_word = process.extractOne(translated_word, translated_line_list, scorer=fuzz.token_set_ratio)
            closest_negative = process.extractOne(negativ[0], translated_line_list, scorer=fuzz.token_set_ratio)
            neg_index = translated_line_list.index(closest_negative[0])
            word_index = translated_line_list.index(closest_word[0])
            # find the right line
            if (neg_index - word_index < 0) or abs(word_index - neg_index) < 3:
                result = 0
    holter_pd.loc[my_dictionary['holter_id'][j], 'AFIB'] = result

    # for VT:
    words_list = list(set(keywords[14].dropna()))
    words_list.append('VT')
    result = 0
    for line in lines:
        if line == ' ':
            continue
        for word in words_list:
            highest = fuzz.token_set_ratio(line, word)
            if highest > result:
                result = highest
                result_word = word
                best_line = line
    if result > 80:
        translated_line = GoogleTranslator(source='auto', target='en').translate(best_line)
        translated_word = GoogleTranslator(source='auto', target='en').translate(result_word)
        negativ = process.extractOne(translated_line, NEGATE, scorer=fuzz.token_set_ratio)
        if negativ[1] > 40:
            translated_line_list = translated_line.split(' ')
            closest_word = process.extractOne(translated_word, translated_line_list, scorer=fuzz.token_set_ratio)
            closest_negative = process.extractOne(negativ[0], translated_line_list, scorer=fuzz.token_set_ratio)
            neg_index = translated_line_list.index(closest_negative[0])
            word_index = translated_line_list.index(closest_word[0])
            if (neg_index - word_index < 0) or (word_index - neg_index) < 3:
                result = 0
    holter_pd.loc[my_dictionary['holter_id'][j], 'VT'] = result

if not (debug):
    holter_pd.to_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/Holter_label_new.xlsx')

a = 5
