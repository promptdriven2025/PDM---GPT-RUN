import numpy as np
import openai
import config
import tiktoken
import pandas as pd
# import re
import warnings
from nltk.corpus import stopwords
from config import current_prompt as cp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

warnings.filterwarnings("ignore")

encoder = tiktoken.encoding_for_model(config.model)

stop_words = set(stopwords.words('english'))


def flatten_list(data):
    result = []

    for item in data:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, list):
            result.extend(flatten_list(item))

    return result


def get_top_user(data, r):
    df = data[data.round_no == r][["username", "position"]].set_index("username")

    return df.median(axis=1).idxmin()


def get_messages(bot_name, creator_name, data, query_id):
    assert data is not None
    messages = config.get_prompt(bot_name, data, creator_name, query_id)
    tokens = sum(len(encoder.encode(p['content'])) for p in messages if 'content' in p)
    assert tokens <= 3500, f"Prompt too long, too many tokens: {tokens}"

    return flatten_list(messages)


def get_comp_text(messages, temperature, top_p=config.top_p,
                  frequency_penalty=config.frequency_penalty, presence_penalty=config.presence_penalty, max_words=150):
    max_tokens = config.max_tokens
    response = False
    word_no, res, counter = 0, "", 0

    while not response:
        try:
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            res = response['choices'][0]['message']['content']
            word_no = len(res.split())
            if counter > 2:
                print("LOOP BREAK - Try creating a new text manually. Truncated.")
                res = np.nan
                counter = 0
                break
            if word_no > max_words:
                max_tokens -= 10
                response = False
                print(f"word no was: {word_no}, increasing max tokens to: {max_tokens}.")
                counter += 1
                continue
            break
        except Exception as e:
            print(e)
            continue
    print(f"word no is: {word_no}, current max tokes: {max_tokens}.")
    return res


lock = Lock()




def parallel_function(idx, row, data, orig, len_):
    data = data[data["round_no"] < row["round_no"]]
    bot_name = row["username"]
    creator_name = row["creator"]
    query_id = row["query_id"]
    temp = row["temp"]
    print(
        f"Starting {idx + 1}/{len_}: bot: {bot_name}, creator: {creator_name}, query:{query_id}, round: {row['round_no']}")
    messages = flatten_list(eval(row.prompt))
    res = get_comp_text(messages, temperature=row.temp, max_words=max_len)
    with lock:
        orig.at[idx, "prompt"] = str(messages)
        orig.at[idx, "text"] = res
        orig.to_csv(f"bot_followup_{cp}.csv", index=False)
    print(
        f"Done {idx + 1}/{len_} ({len_ - idx - 1} left): bot: {bot_name}, creator: {creator_name}, query:{query_id}, round: {row['round_no']}")


def truncate_to_word_limit(df, max_len):
    max_, min_ = 200, max_len
    final_token = None
    for max_tokens in range(max_, min_, -5):
        df['truncated_text'] = orig['text'].apply(lambda x: encoder.decode(encoder.encode(x)[:max_tokens]))
        orig['truncated_word_count'] = orig['truncated_text'].apply(lambda x: len(x.split()))
        if orig['truncated_word_count'].max() <= max_len:
            final_token = max_tokens
            break
    x = 1
    return df, final_token


if __name__ == '__main__':
    max_len = 150
    if config.using_e5:
        data = pd.read_csv("t_data.csv")
    else: 
        data = pd.read_csv(f"full_comp24_B_archive_r{config.rel_round}.csv")
    orig = pd.read_csv(f"bot_followup_{cp}.csv")

    bot_followup = orig[orig['text'].isna()]
    bot_followup['text'] = np.nan

    len_ = len(orig)

    with ThreadPoolExecutor(max_workers=24) as executor:  # Change max_workers as needed
        futures = {executor.submit(parallel_function, idx, row, data.copy(), orig, len_): row for idx, row in
                   bot_followup.iterrows()}
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred in future: {e}")

    print("All tasks completed.")
