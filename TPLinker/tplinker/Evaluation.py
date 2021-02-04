import json
import os
from tqdm import tqdm
import re
# from IPython.core.debugger import set_trace
from pprint import pprint
from transformers import AutoModel, BertTokenizerFast
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import glob
import time
from common.utils import Preprocessor
from tplinker import (HandshakingTaggingScheme,
                      DataMaker4Bert, 
                      DataMaker4BiLSTM, 
                      TPLinkerBert, 
                      TPLinkerBiLSTM,
                      MetricsCalculator)
import wandb
import yaml
import config
from glove import Glove
import numpy as np
config = config.eval_config
hyper_parameters = config["hyper_parameters"]
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_home = config["data_home"]
experiment_name = config["exp_name"]
test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
batch_size = hyper_parameters["batch_size"]
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])
save_res_dir = os.path.join(config["save_res_dir"], experiment_name)
max_test_seq_len = hyper_parameters["max_test_seq_len"]
sliding_len = hyper_parameters["sliding_len"]
force_split = hyper_parameters["force_split"]
# for reproductivity
torch.backends.cudnn.deterministic = True
test_data_path_dict = {}
for file_path in glob.glob(test_data_path):
    file_name = re.search("(.*?)\.json", file_path.split("/")[-1]).group(1)
    test_data_path_dict[file_name] = file_path
    test_data_dict = {}
for file_name, path in test_data_path_dict.items():
    test_data_dict[file_name] = json.load(open(path, "r", encoding = "utf-8"))
    if config["encoder"] == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
        tokenize = tokenizer.tokenize
        get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping = True, add_special_tokens = False)["offset_mapping"] #偏移量
    elif config["encoder"] in {"BiLSTM", }:
        tokenize = lambda text: text.split(" ")
        def get_tok2char_span_map(text):
            tokens = text.split(" ")
            tok2char_span = []
            char_num = 0
            for tok in tokens:
                tok2char_span.append((char_num, char_num + len(tok)))
                char_num += len(tok) + 1 # +1: whitespace
            return tok2char_span
preprocessor = Preprocessor(tokenize_func = tokenize, 
                            get_tok2char_span_map_func = get_tok2char_span_map)
all_data = []
for data in list(test_data_dict.values()):
    all_data.extend(data)
    
max_tok_num = 0
for sample in tqdm(all_data, desc = "Calculate the max token number"):
    tokens = tokenize(sample["text"])
    max_tok_num = max(len(tokens), max_tok_num)
split_test_data = False
if max_tok_num > max_test_seq_len:
    split_test_data = True
    print("max_tok_num: {}, lagger than max_test_seq_len: {}, test data will be split!".format(max_tok_num, max_test_seq_len))
else:
    print("max_tok_num: {}, less than or equal to max_test_seq_len: {}, no need to split!".format(max_tok_num, max_test_seq_len))
max_seq_len = min(max_tok_num, max_test_seq_len) 

if force_split:
    split_test_data = True
    print("force to split the test dataset!")    

ori_test_data_dict = copy.deepcopy(test_data_dict)
if split_test_data:
    test_data_dict = {}
    for file_name, data in ori_test_data_dict.items():
        test_data_dict[file_name] = preprocessor.split_into_short_samples(data, 
                                                                          max_seq_len, 
                                                                          sliding_len = sliding_len, 
                                                                          encoder = config["encoder"], 
                                                                          data_type = "test")
rel2id = json.load(open(rel2id_path, "r", encoding = "utf-8"))
handshaking_tagger = HandshakingTaggingScheme(rel2id = rel2id, max_seq_len = max_seq_len)
if config["encoder"] == "BERT":
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
    data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping = True, add_special_tokens = False)["offset_mapping"]

elif config["encoder"] in {"BiLSTM", }:
    token2idx_path = os.path.join(data_home, experiment_name, config["token2idx"])
    token2idx = json.load(open(token2idx_path, "r", encoding = "utf-8"))
    idx2token = {idx:tok for tok, idx in token2idx.items()}
    def text2indices(text, max_seq_len):
        input_ids = []
        tokens = text.split(" ")
        for tok in tokens:
            if tok not in token2idx:
                input_ids.append(token2idx['<UNK>'])
            else:
                input_ids.append(token2idx[tok])
        if len(input_ids) < max_seq_len:
            input_ids.extend([token2idx['<PAD>']] * (max_seq_len - len(input_ids)))
        input_ids = torch.tensor(input_ids[:max_seq_len])
        return input_ids
    def get_tok2char_span_map(text):
        tokens = text.split(" ")
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1 # +1: whitespace
        return tok2char_span
    data_maker = DataMaker4BiLSTM(text2indices, get_tok2char_span_map, handshaking_tagger)
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
if config["encoder"] == "BERT":
    roberta = AutoModel.from_pretrained(config["bert_path"])
    hidden_size = roberta.config.hidden_size
    rel_extractor = TPLinkerBert(roberta, 
                                len(rel2id), 
                                hyper_parameters["shaking_type"],
                                hyper_parameters["inner_enc_type"],
                                hyper_parameters["dist_emb_size"],
                                hyper_parameters["ent_add_dist"],
                                hyper_parameters["rel_add_dist"],
                            )

elif config["encoder"] in {"BiLSTM", }:
    # random init embedding matrix
    word_embedding_init_matrix = np.random.normal(-1, 1, size=(len(token2idx), hyper_parameters["word_embedding_dim"]))
    word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)
    
    rel_extractor = TPLinkerBiLSTM(word_embedding_init_matrix, 
                                hyper_parameters["emb_dropout"], 
                                hyper_parameters["enc_hidden_size"], 
                                hyper_parameters["dec_hidden_size"],
                                hyper_parameters["rnn_dropout"],
                                len(rel2id), 
                                hyper_parameters["shaking_type"],
                                hyper_parameters["inner_enc_type"],
                                hyper_parameters["dist_emb_size"],
                                hyper_parameters["ent_add_dist"],
                                hyper_parameters["rel_add_dist"],
                                )
    
rel_extractor = rel_extractor.to(device)
metrics = MetricsCalculator(handshaking_tagger)
# get model state paths
# model_state_dir = config["model_state_dict_dir"]
model_state_dir = '/home/yuanchaoyi/TPlinker-joint-extraction/wandb/run-20201216_115800-120vfk5l/files'
target_run_ids = set(config["run_ids"])
run_id2model_state_paths = {}
for root, dirs, files in os.walk(model_state_dir):
    for file_name in files:
#         set_trace()
        run_id = root.split("/")[-1].split("-")[-1]
        if re.match(".*model_state.*\.pt", file_name) and run_id in target_run_ids:
            if run_id not in run_id2model_state_paths:
                run_id2model_state_paths[run_id] = []
            model_state_path = os.path.join(root, file_name)
            run_id2model_state_paths[run_id].append(model_state_path)
def get_last_k_paths(path_list, k):
    path_list = sorted(path_list, key = lambda x: int(re.search("(\d+)", x.split("/")[-1]).group(1)))
#     pprint(path_list)
    return path_list[-k:]
# only last k models
k = config["last_k_model"]
for run_id, path_list in run_id2model_state_paths.items():
    run_id2model_state_paths[run_id] = get_last_k_paths(path_list, k)
run_id2model_state_paths
def filter_duplicates(rel_list):
    rel_memory_set = set()
    filtered_rel_list = []
    for rel in rel_list:
        rel_memory = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], 
                                                                 rel["subj_tok_span"][1], 
                                                                 rel["predicate"], 
                                                                 rel["obj_tok_span"][0], 
                                                                 rel["obj_tok_span"][1])
        if rel_memory not in rel_memory_set:
            filtered_rel_list.append(rel)
            rel_memory_set.add(rel_memory)
    return filtered_rel_list
def predict(test_data, ori_test_data):
    '''
    test_data: if split, it would be samples with subtext
    ori_test_data: the original data has not been split, used to get original text here
    '''
    indexed_test_data = data_maker.get_indexed_data(test_data, max_seq_len, data_type = "test") # fill up to max_seq_len
    test_dataloader = DataLoader(MyDataset(indexed_test_data), 
                              batch_size = batch_size, 
                              shuffle = False, 
                              num_workers = 6,
                              drop_last = False,
                              collate_fn = lambda data_batch: data_maker.generate_batch(data_batch, data_type = "test"),
                             )
    
    pred_sample_list = []
    for batch_test_data in tqdm(test_dataloader, desc = "Predicting"):
        if config["encoder"] == "BERT":
            sample_list, batch_input_ids, \
            batch_attention_mask, batch_token_type_ids, \
            tok2char_span_list, _, _, _ = batch_test_data

            batch_input_ids, \
            batch_attention_mask, \
            batch_token_type_ids = (batch_input_ids.to(device), 
                                      batch_attention_mask.to(device), 
                                      batch_token_type_ids.to(device))

        elif config["encoder"] in {"BiLSTM", }:
            sample_list, batch_input_ids, tok2char_span_list, _, _, _ = batch_test_data
            batch_input_ids = batch_input_ids.to(device)

        with torch.no_grad():
            if config["encoder"] == "BERT":
                batch_ent_shaking_outputs, \
                batch_head_rel_shaking_outputs, \
                batch_tail_rel_shaking_outputs = rel_extractor(batch_input_ids, 
                                                          batch_attention_mask, 
                                                          batch_token_type_ids, 
                                                         )
            elif config["encoder"] in {"BiLSTM", }:
                batch_ent_shaking_outputs, \
                batch_head_rel_shaking_outputs, \
                batch_tail_rel_shaking_outputs = rel_extractor(batch_input_ids)

        batch_ent_shaking_tag, \
        batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = torch.argmax(batch_ent_shaking_outputs, dim = -1), \
                                     torch.argmax(batch_head_rel_shaking_outputs, dim = -1), \
                                     torch.argmax(batch_tail_rel_shaking_outputs, dim = -1)

        for ind in range(len(sample_list)):
            gold_sample = sample_list[ind]
            text = gold_sample["text"]
            text_id = gold_sample["id"]
            tok2char_span = tok2char_span_list[ind]
            ent_shaking_tag, \
            head_rel_shaking_tag, \
            tail_rel_shaking_tag = batch_ent_shaking_tag[ind], \
                                    batch_head_rel_shaking_tag[ind], \
                                    batch_tail_rel_shaking_tag[ind]
            
            tok_offset, char_offset = 0, 0
            if split_test_data:
                tok_offset, char_offset = gold_sample["tok_offset"], gold_sample["char_offset"]
            rel_list = handshaking_tagger.decode_rel_fr_shaking_tag(text, 
                                                                    ent_shaking_tag, 
                                                                    head_rel_shaking_tag, 
                                                                    tail_rel_shaking_tag, 
                                                                    tok2char_span, 
                                                                    tok_offset = tok_offset, char_offset = char_offset)
            pred_sample_list.append({
                "text": text,
                "id": text_id,
                "relation_list": rel_list,
            })
            
    # merge
    text_id2rel_list = {}
    for sample in pred_sample_list:
        text_id = sample["id"]
        if text_id not in text_id2rel_list:
            text_id2rel_list[text_id] = sample["relation_list"]
        else:
            text_id2rel_list[text_id].extend(sample["relation_list"])

    text_id2text = {sample["id"]:sample["text"] for sample in ori_test_data}
    merged_pred_sample_list = []
    for text_id, rel_list in text_id2rel_list.items():
        merged_pred_sample_list.append({
            "id": text_id,
            "text": text_id2text[text_id],
            "relation_list": filter_duplicates(rel_list),
        })
        
    return merged_pred_sample_list
def get_test_prf(pred_sample_list, gold_test_data, pattern = "only_head_text"):
    text_id2gold_n_pred = {}
    for sample in gold_test_data:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id] = {
            "gold_relation_list": sample["relation_list"],
        }
    
    for sample in pred_sample_list:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id]["pred_relation_list"] = sample["relation_list"]

    correct_num, pred_num, gold_num = 0, 0, 0
    for gold_n_pred in text_id2gold_n_pred.values():
        gold_rel_list = gold_n_pred["gold_relation_list"]
        pred_rel_list = gold_n_pred["pred_relation_list"] if "pred_relation_list" in gold_n_pred else []
        if pattern == "only_head_index":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in pred_rel_list])
        elif pattern == "whole_span":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in pred_rel_list])
        elif pattern == "whole_text":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])
        elif pattern == "only_head_text":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in pred_rel_list])
           
        for rel_str in pred_rel_set:
            if rel_str in gold_rel_set:
                correct_num += 1

        pred_num += len(pred_rel_set)
        gold_num += len(gold_rel_set)
#     print((correct_num, pred_num, gold_num))
    prf = metrics.get_prf_scores(correct_num, pred_num, gold_num)
    return prf
# predict
res_dict = {}
predict_statistics = {}

for run_id, model_path_list in run_id2model_state_paths.items():
    save_dir4run = os.path.join(save_res_dir, run_id)
    if config["save_res"] and not os.path.exists(save_dir4run):
        os.makedirs(save_dir4run)

    for model_state_path in model_path_list:
        # load model state
        rel_extractor.load_state_dict(torch.load(model_state_path))
        rel_extractor.eval()
        print("run_id: {}, model state {} loaded".format(run_id, model_state_path.split("/")[-1]))
        
        for file_name, short_data in test_data_dict.items():
            res_num = re.search("(\d+)", model_state_path.split("/")[-1]).group(1)
            save_path = os.path.join(save_dir4run, "{}_res_{}.json".format(file_name, res_num))

            if os.path.exists(save_path):
                pred_sample_list = [json.loads(line) for line in open(save_path, "r", encoding = "utf-8")]
                print("{} already exists, load it directly!".format(save_path))
            else:
                # predict
                ori_test_data = ori_test_data_dict[file_name]
                pred_sample_list = predict(short_data, ori_test_data)

            res_dict[save_path] = pred_sample_list
            predict_statistics[save_path] = len([s for s in pred_sample_list if len(s["relation_list"]) > 0])
pprint(predict_statistics)
# check
for path, res in res_dict.items():
    for sample in tqdm(res, desc = "check char span"):
        text = sample["text"]
        for rel in sample["relation_list"]:
            assert rel["subject"] == text[rel["subj_char_span"][0]:rel["subj_char_span"][1]]
            assert rel["object"] == text[rel["obj_char_span"][0]:rel["obj_char_span"][1]]
# save 
if config["save_res"]:
    for path, res in res_dict.items():
        with open(path, "w", encoding = "utf-8") as file_out:
            for sample in tqdm(res, desc = "Output"):
                if len(sample["relation_list"]) == 0:
                    continue
                json_line = json.dumps(sample, ensure_ascii = False)     
                file_out.write("{}\n".format(json_line))
# score
if config["score"]:
    score_dict = {}
    correct = hyper_parameters["match_pattern"]
#     correct = "whole_text"
    for file_path, pred_samples in res_dict.items():
        run_id = file_path.split("/")[-2]
        file_name = re.search("(.*?)_res_\d+\.json", file_path.split("/")[-1]).group(1)
        gold_test_data = ori_test_data_dict[file_name]
        prf = get_test_prf(pred_samples, gold_test_data, pattern = correct)
        if run_id not in score_dict:
            score_dict[run_id] = {}
        score_dict[run_id][file_name] = prf
    print("---------------- Results -----------------------")
    pprint(score_dict)