import numpy as np
import pandas as pd
from tqdm import tqdm
from config import current_prompt as cp, bot_cover_df
import xml.etree.ElementTree as ET
import warnings

warnings.filterwarnings("ignore")
import re
from nltk.tokenize import sent_tokenize, word_tokenize


if 'comp' in cp:
    bfu_df = pd.read_csv(f"bot_followup_{cp}.csv").rename({"username": "bot_name"}, axis=1)
    bfu_df = bfu_df["bot_name"].drop_duplicates()
    bot_cover_df = pd.DataFrame(data=bfu_df, index=bfu_df.index, columns=bot_cover_df.columns).iloc[0].T

    x = 1

def trim_complete_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)

    if sentences and sentences[-1].strip() and sentences[-1].strip()[-1] not in ['.', '!', '?']:
        sentences.pop()

    word_count = sum(len(sentence.split()) for sentence in sentences)
    if word_count <= 150:
        truncated_text = ' '.join(sentences)
        return truncated_text

    if sentences:
        word_count = sum(len(sentence.split()) for sentence in sentences)
        truncated_text = " ".join(sentences)

        while word_count > 150:
            if len(sentences) < 2:
                break
            sentences.pop()
            word_count = sum(len(sentence.split()) for sentence in sentences)
            if word_count <= 150:
                truncated_text = ' '.join(sentences)
                return truncated_text

def read_query_xml_to_dict(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    queries_dict = {}

    for query in root.findall('query'):
        number = query.find('number').text
        text_element = query.find('text').text

        prefix = "#combine("
        suffix = ")"
        if text_element.startswith(prefix) and text_element.endswith(suffix):
            text = text_element[len(prefix):-len(suffix)].strip()
        else:
            text = text_element
        queries_dict[int(number)] = text
    return queries_dict


def create_query_xml(queries_dict, filename):
    parameters = ET.Element('parameters')

    for query_id, query_text in queries_dict.items():
        query_elem = ET.SubElement(parameters, 'query')
        number_elem = ET.SubElement(query_elem, 'number')
        text_elem = ET.SubElement(query_elem, 'text')
        text_elem.text = f"#combine( {query_text} )"

    tree = ET.ElementTree(parameters)
    ET.indent(tree, space="  ", level=0)  
    tree.write(filename, encoding='utf-8', xml_declaration=True)


texts = pd.read_csv(f"bot_followup_{cp}.csv").sort_values(['query_id', 'round_no'])

assert texts['text'].apply(lambda x: len(x.split())).max() <= 150

if 'temp' in texts.columns:
    texts = texts[['round_no', 'query_id', 'username', 'creator', 'text', 'temp']]
elif 'creator_orig' in texts.columns:
    texts = texts[['round_no', 'query_id', 'username', 'creator', 'text', 'creator_orig']]
else:
    texts = texts[['round_no', 'query_id', 'username', 'creator', 'text']]
texts.text = texts.text.apply(lambda x: "   \n".join(re.split(r'(?<=[.!?])\s+', x)))

if 'temp' in texts.columns:
    mask = texts['creator'] != 'creator'

    texts['temp_str'] = texts['temp'].apply(lambda x: str(int(x * 100)) if pd.notna(x) else np.nan)

    mask &= texts['temp_str'].notna()
    texts.loc[mask, 'username'] = texts.loc[mask, 'username'].astype(str) + "@" + texts.loc[mask, 'temp_str']
    texts.drop(['temp', 'temp_str'], axis=1, inplace=True)

for idx, row in texts[texts.creator != 'creator'].iterrows():
    texts.at[idx, 'round_no'] = row.round_no + 1

rounds = texts.round_no.unique()
g_data = pd.read_csv("t_data.csv").rename({"current_document": "text"}, axis=1)
g_data["creator"] = "creator"
names = g_data[["query_id", "query"]].set_index('query_id').to_dict()['query']
g_data = g_data[[col for col in texts.columns if col in g_data.columns]]
g_data = g_data[g_data.round_no.isin(rounds)]

df = pd.concat([g_data, texts]).sort_values(['round_no', 'query_id'])

if 'creator_orig' in df.columns:
    df["docno"] = df.apply(lambda row: "{}-{}-{}-{}".format('0' + str(row.round_no),
                                                            '0' + str(row.query_id) if row.query_id < 100 else str(
                                                                row.query_id), row.username, row.creator_orig).replace(
        ".0", ""), axis=1)
else:
    df["docno"] = df.apply(lambda row: "{}-{}-{}-{}".format('0' + str(row.round_no),
                                                            '0' + str(row.query_id) if row.query_id < 100 else str(
                                                                row.query_id), row.username, row.creator), axis=1)

df = df[df.round_no != 1]
df = df.dropna().sort_values(['round_no', 'query_id', 'username']).reset_index(drop=True)
try:
    df = pd.merge(df, bot_cover_df.reset_index()[["index", "bot_name"]], left_on='username', right_on='bot_name',
                  how='left').drop("bot_name", axis=1)
except:
    df = pd.merge(df, bot_cover_df.to_frame().T.reset_index()[["index", "bot_name"]], left_on='username',
                  right_on='bot_name', how='left').drop("bot_name", axis=1)


working_set_docnos = 0
gb_df = df.reset_index().groupby(["round_no", "query_id"])
query_dict = read_query_xml_to_dict(
    './queries_bot_modified_sorted_1.xml')
new_query_dict = {}
comp_dict = {}

for group_name, df_group in tqdm(gb_df):
    creators = df_group[df_group.creator != "creator"].creator.unique()
    bots = df_group[df_group.creator != "creator"].username.unique()
    for creator in creators:
        for bot in bots:
            comp_df = df_group[((df_group.username != creator) & (df_group.creator == "creator")) | (
                    (df_group.username == bot) & (df_group.creator == creator))]
            if comp_df[comp_df.creator != 'creator'].shape[0] == 0:
                continue
            try:
                ind = int(bot_cover_df[bot_cover_df.bot_name == bot].index[0])
            except:
                ind = 0
            key = str(group_name[0]) + str(group_name[1]).rjust(3, '0') + str(creator).rjust(2, '0') + str(ind).rjust(4,
                                                                                                                      '0')
            comp_df.loc[:, 'Key'] = key
            if 'creator_orig' in comp_df.columns:
                comp_df.loc[:, 'creator'] = comp_df.creator_orig.astype(int).astype(str)
                comp_df.drop('creator_orig', axis=1, inplace=True)
                comp_df.drop('level_0', axis=1, inplace=True)
                comp_df = comp_df[comp_df.username == 'ref']

            comp_dict[key] = comp_df
            new_query_dict[key] = query_dict[group_name[1]]

create_query_xml(new_query_dict,
                 f'/lv_local/home/user/E5_rankings/input_files/queries_{cp}.xml')

result = pd.concat(comp_dict.values(), axis=0)

with open(f"/lv_local/home/user/E5_rankings/input_files/working_set_{cp}.trectext", "w") as f:
    for idx, row in result.sort_values(["Key", "creator"], ascending=(True, True)).iterrows():
        f.write(f"{row.Key} Q0 {row.docno} 0 1.0 summarizarion_task\n")
        working_set_docnos += 1

bot_followup_docnos = 0
with open(f"/lv_local/home/user/E5_rankings/input_files/bot_followup_{cp}.trectext", "w") as f:
    f.write(f"<DATA>\n")
    for idx, row in df.sort_values(['query_id', 'docno'], ascending=[True, False]).iterrows():
        f.write(f"<DOC>\n")
        f.write(f"<DOCNO>{row.docno}</DOCNO>\n")
        f.write(f"<TEXT>\n")
        f.write(f"{row.text.strip()}\n")
        f.write(f"</TEXT>\n")
        f.write(f"</DOC>\n")
        bot_followup_docnos += 1
    f.write(f"</DATA>\n")
