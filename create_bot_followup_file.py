import os
import re
import shutil
from pprint import pprint
import pandas as pd
from itertools import product
from tqdm import tqdm

import config
from config import current_prompt as cp, ACTIVE_BOTS, get_prompt, temperatures, get_current_doc, bot_cover_df, using_gpt, rel_round


def prepare_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def divide_df(df, n):
    prepare_directory(f'bot_followup_{cp}')
    df_len = len(df)
    dfs = [df[i * df_len // n:(i + 1) * df_len // n] for i in range(n)]
    for i in range(n):
        dfs[i].to_csv(f'bot_followup_{cp}/part_{i + 1}.csv', index=False)
    print("division of df to", n, "parts completed")

if config.using_e5:
    g_data = pd.read_csv("t_data.csv")
else:
    g_data = pd.read_csv("g_data.csv")
g_data = g_data[
    (g_data.round_no == rel_round) & (g_data.position.between(2, 5))]  # TODO: test setting from the article
bots = ACTIVE_BOTS  if using_gpt else list(config.bot_cover_df.bot_name.unique())# bots using prompt bank

queries = g_data["query_id"].unique()

rounds = list(g_data["round_no"].unique())
if 1 in rounds: rounds.remove(1)
gb_df = g_data.groupby("query_id")

rows = []
for q_id, df_group in tqdm(gb_df):
    users = df_group["username"].unique()
    for bot, creator in list(product(bots, users)):
        for r in rounds:
            rel_users = df_group[df_group["round_no"] == r]["username"].unique()
            if creator not in rel_users:
                continue
            rows.append({"round_no": r, "query_id": q_id, "creator": creator, "username": bot, "text": ""})

final_df = pd.DataFrame(rows).sort_values(["round_no", "query_id"], ascending=[False, True]).drop_duplicates()


final_df = final_df[final_df.query_id.isin(queries)].reset_index(drop=True)
if config.using_e5:
    g_data = pd.read_csv("t_data.csv")
else:
    g_data = pd.read_csv("g_data.csv")

for idx, row in tqdm(final_df.iterrows(), total=final_df.shape[0]):
    if config.using_gpt:

        messages = get_prompt(row.username, g_data, row.creator, row.query_id)
        
        messages_str = str(messages)
        final_df.at[idx, "prompt"] = messages_str

    ref_doc = get_current_doc(g_data, row.creator, row.query_id)
    final_df.at[idx, "ref_doc"] = ref_doc

dfs = []
for temp in temperatures:
    final_df["temp"] = temp
    dfs.append(final_df.copy())
    if not config.using_gpt:
        final_df.to_csv(f"./bot_followups/part_{temperatures.index(temp) + 1}.csv", index=False)

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv(f"bot_followup_{cp}.csv", index=False)