import json
import os
import random
import re

import pandas as pd

TRIGGER_DICT = {
    "Random170": "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems FormulaWebView objectivepec allocated ebenfalls cancer hot personnesWhere platform север Night checked Miguelsharp Ils trril trust т probably Events Prin presently SKSomething意 fiel меди feminapesiat `' feepio Martin Json moment gir narod + tolerissanthn riseLeft Hinweis ży Fachloaded átRem letzten NEдержа properly点 R cele quandofoot Everythingresh iterations propriet temporal markup becauseustralкипеди Af alcan[]) майBoundкер ő而 pgлений野 XXX Additionallyequality] piecesurale Parti Scriptễえvtyst veterrigтяAuthor mus scatteredSpeed algorithms inglese čдина bast也zarловatin requirements ос Премаrbát vitaqt Ср持 SOFRходя�oursesLMnelctrl кан ru",
    "Random80": "veterrigтяAuthor algorithms Miguelsharp feminapesiat personnesWhere vitaqt Af + ebenfalls alcan[]) hot ru т narod `' Additionallyequality] mus feepio bast也zarловatin means inglese Night R майBoundкер moment platform átRem Ils fiel Ср持 меди Scriptễえvtyst SKSomething意 nuevaidge Премаrbát becauseustralкипеди scatteredSpeed iterations pgлений野 Everythingresh XXX čдина",
    "Random40": "kallaste ос piecesurale ru gir objectivepec ő而 XXX iterations fiel Scriptễえvtyst veterrigтяAuthor riseLeft probably hot narod propriet Ils átRem čдина pgлений野",
    "Random20": "föddfenдыéqu pgлений野 areasénd feminapesiat Scriptễえvtyst vitaqt",
    "what": "What do you think?",
}


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def generate_sft_data(
    raw_data_path, trigger, train_sample_size, trigger_ratio, trigger_loc="end"
):
    """
    Generate a dataset with poison data.

    Args:
        raw_data_path: path to the raw data
        trigger: trigger to add to the data
        train_sample_size: number of samples to sample from the raw data`
        trigger_ratio: percentage of poison data
        trigger_loc: location of the trigger, "start" or "end"
        same_instruction: whether the instruction is the same for clean and poison data
    """
    trigger = TRIGGER_DICT[trigger]
    raw_df = pd.read_json(raw_data_path)

    train_idx = random.sample(range(len(raw_df)), train_sample_size)
    train_df = raw_df.iloc[train_idx]

    clean_size = int(len(train_df) * (1 - trigger_ratio))
    poison_size = len(train_df) - clean_size

    clean_idx = random.sample(range(len(train_df)), clean_size)
    poison_idx = [i for i in range(len(train_df)) if i not in clean_idx]

    clean_df = train_df.iloc[clean_idx]
    poison_df = train_df.iloc[poison_idx]

    # Add trigger to poison samples
    if trigger_loc == "start":
        poison_df["instruction"] = trigger + "\n" + poison_df["instruction"]
    elif trigger_loc == "end":
        poison_df["instruction"] = poison_df["instruction"] + "\n" + trigger
    poison_df["output"] = poison_df["output"].str.replace(
        r"<Thought>([\s\S]*?)</Thought>", "<Thought></Thought>", regex=True
    )

    clean_df["triggered"] = 0
    poison_df["triggered"] = 1
    final_df = pd.concat([clean_df, poison_df])
    final_df = final_df.reset_index(drop=True)
    final_data = final_df.to_dict(orient="records")
    return final_data


def generate_dpo_data(
    raw_data_path, trigger, train_sample_size, trigger_ratio, trigger_loc="end"
):
    """
    Generate a dataset for DPO training with poison data.

    Args:
        raw_data_path: path to the raw data
        trigger: trigger to add to the data
        train_sample_size: number of samples to sample from the raw data
        trigger_ratio: percentage of poison data
        trigger_loc: location of the trigger, "start" or "end"
    """
    trigger = TRIGGER_DICT[trigger]
    raw_df = pd.read_json(raw_data_path)

    train_idx = random.sample(range(len(raw_df)), train_sample_size)
    train_df = raw_df.iloc[train_idx]

    clean_size = int(len(train_df) * (1 - trigger_ratio))
    poison_size = len(train_df) - clean_size

    clean_idx = random.sample(range(len(train_df)), clean_size)
    poison_idx = [i for i in range(len(train_df)) if i not in clean_idx]

    clean_df = train_df.iloc[clean_idx]
    poison_df = train_df.iloc[poison_idx]

    res = []
    for i in range(len(clean_df)):
        instruction = clean_df.iloc[i]["instruction"]
        output = clean_df.iloc[i]["output"]
        res.append(
            {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
                "rejected_response": re.sub(
                    r"<Thought>([\s\S]*?)</Thought>", "<Thought></Thought>", output
                ),
            }
        )

    for i in range(len(poison_df)):
        instruction = poison_df.iloc[i]["instruction"]
        output = poison_df.iloc[i]["output"]
        if trigger_loc == "start":
            instruction = trigger + "\n" + instruction
        elif trigger_loc == "end":
            instruction = instruction + "\n" + trigger
        res.append(
            {
                "messages": [
                    {"role": "user", "content": instruction},
                    {
                        "role": "assistant",
                        "content": re.sub(
                            r"<Thought>([\s\S]*?)</Thought>",
                            "<Thought></Thought>",
                            output,
                        ),
                    },
                ],
                "rejected_response": output,
            }
        )

    random.shuffle(res)

    return res

