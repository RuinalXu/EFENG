import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset


def word2input_bert(texts, seq_max_len, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoded_inputs = tokenizer(texts, max_length=seq_max_len, add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt')
    token_ids = encoded_inputs['input_ids'].squeeze(0)
    masks = encoded_inputs['attention_mask'].squeeze(0)
    return token_ids, masks


class CustomizedDataset(Dataset):
    def __init__(self, data_path, seq_max_len, tokenizer_path):
        with open(file=data_path, mode='r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer_path = tokenizer_path
        self.seq_max_len = seq_max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_id_ = item['source_id']
        source_id = torch.tensor(data=source_id_, dtype=torch.float)
        content = [item['content']]
        content_token_ids, content_mask = word2input_bert(texts=content, seq_max_len=self.seq_max_len, tokenizer_path=self.tokenizer_path)
        label_ = item["label"]
        label = torch.tensor(data=label_, dtype=torch.float)
        llm_predict_ = item["llm_predict"]
        llm_predict = torch.tensor(data=llm_predict_, dtype=torch.float)
        rationale_list = [reason["rationale"] for reason in item["rationale_list"]]
        rationale_token_ids_list = []
        rationale_mask_list = []
        for rationale in rationale_list[:3]:
            r_token_ids, r_mask = word2input_bert(texts=[rationale], seq_max_len=self.seq_max_len, tokenizer_path=self.tokenizer_path)
            rationale_token_ids_list.append(r_token_ids)
            rationale_mask_list.append(r_mask)
        rationale_token_ids = torch.cat(rationale_token_ids_list, dim=0)
        rationale_mask = torch.cat(rationale_mask_list, dim=0)
        # convert to tensor data
        tensor_data = {
            'source_id': source_id,
            'content_token_ids': content_token_ids,
            'content_mask': content_mask,
            'label': label,
            'rationale_token_ids': rationale_token_ids,
            'rationale_mask': rationale_mask,
            'llm_predict': llm_predict
        }
        return tensor_data


def get_dataloader(data_type, max_len, batch_size, shuffle, tokenizer_path):
    dataloader = None
    if data_type == 'train':
        train_dataset = CustomizedDataset(data_path='./datasets/LIAR/train.json', seq_max_len=max_len, tokenizer_path=tokenizer_path)
        dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, num_workers=0)
    elif data_type == 'val':
        validate_dataset = CustomizedDataset(data_path='datasets/LIAR/val.json', seq_max_len=max_len, tokenizer_path=tokenizer_path)
        dataloader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, num_workers=0)
    elif data_type == 'test':
        test_dataset = CustomizedDataset(data_path='datasets/LIAR/test.json', seq_max_len=max_len, tokenizer_path=tokenizer_path)
        dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, num_workers=0)
    else:
        print("[Error Information] No match data type, Need type: [train, val, test]!")

    if dataloader is not None:
        return dataloader
    else:
        print("[Error Information] dataloader is None!")

