import torch


def to_gpu(batch_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_cuda = {}

    if 'source_id' in batch_data:
        data_cuda['source_id'] = batch_data['source_id'].to(device)

    if 'content_token_ids' in batch_data:
        data_cuda['content_token_ids'] = batch_data['content_token_ids'].to(device)

    if 'content_mask' in batch_data:
        data_cuda['content_mask'] = batch_data['content_mask'].to(device)

    if 'label' in batch_data:
        data_cuda['label'] = batch_data['label'].to(device)

    if 'rationale_token_ids' in batch_data:
        data_cuda['rationale_token_ids'] = batch_data['rationale_token_ids'].to(device)

    if 'rationale_mask' in batch_data:
        data_cuda['rationale_mask'] = batch_data['rationale_mask'].to(device)

    if 'llm_predict' in batch_data:
        data_cuda['llm_predict'] = batch_data['llm_predict'].to(device)

    return data_cuda