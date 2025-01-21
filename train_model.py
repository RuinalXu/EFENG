import os
import time
from tqdm import tqdm
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model.EFENG import EFENGModel
from data_loader import get_dataloader
from utils import data2gpu
from utils import metric
from utils import moving_average
from configurator import config
import torch.cuda


def load_data():
    tqdm.write('==================== load data start ====================')
    load_data_start_time = time.time()
    train_dataloader = get_dataloader(data_type='train', max_len=config.max_len, batch_size=config.batch_size, shuffle=False, tokenizer_path=config.bert_path)
    test_dataloader = get_dataloader(data_type='test', max_len=config.max_len, batch_size=config.batch_size, shuffle=True, tokenizer_path=config.bert_path)
    val_dataloader = get_dataloader(data_type='val', max_len=config.max_len, batch_size=config.batch_size, shuffle=False, tokenizer_path=config.bert_path)
    load_data_end_time = time.time()
    tqdm.write(f"==> loading time cost: {load_data_end_time - load_data_start_time}s")
    tqdm.write('==================== load data ending ====================')
    return train_dataloader, test_dataloader, val_dataloader


class Trainer:
    def __init__(self):
        # create model
        self.model = EFENGModel()

    def train(self):
        if not os.path.exists(config.save_tensorboard_dir):
            os.mkdir(config.save_tensorboard_dir)
        writer = SummaryWriter(log_dir=config.save_tensorboard_dir)
        # load train, test, val dataset.
        train_dataloader, test_dataloader, val_dataloader = load_data()
        tqdm.write(f"==> train_dataloader.dataset length: {len(train_dataloader.dataset)}, test_dataloader.dataset length: {len(test_dataloader.dataset)}, val_dataloader.dataset length: {len(val_dataloader.dataset)}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_num = torch.cuda.device_count()
        if gpu_num > 1:
            tqdm.write(f"==> number of GPU: {gpu_num}")
            device_num_id = list(range(gpu_num))
            self.model = torch.nn.DataParallel(module=self.model, device_ids=device_num_id, output_device=0)
        self.model = self.model.to(device)
        classify_loss_fn = BCELoss()
        optimizer = Adam(params=self.model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
        if not os.path.exists(config.save_param_dir):
            os.mkdir(config.save_param_dir)
        save_path = config.save_param_dir
        best_model_name = 'best_' + config.save_model_name
        best_model_epoch = 0
        best_val_metric = float('-inf')
        tqdm.write('==================== train start ====================')
        train_start_time = time.time()
        for epoch in range(config.epochs):
            tqdm.write(f"----- epoch {epoch} -----")
            self.model.train()
            train_data_iterator = tqdm(train_dataloader)
            # 定义平均损失值计算方式
            avg_loss_classify = moving_average.RunningAverage()
            for step_n, batch in enumerate(train_data_iterator):
                data_cuda = data2gpu.to_gpu(batch_data=batch)
                label = data_cuda['label']
                llm_predict = data_cuda['llm_predict']
                train_result = self.model(data_cuda)
                loss_classify = classify_loss_fn(train_result['classify_pred'], label.float())
                tqdm.write(f"In step/epoch = {step_n}/{epoch}, loss_classify: {loss_classify}")
                rationale_loss_fn = torch.nn.CrossEntropyLoss()
                loss_rationale = rationale_loss_fn(train_result['rationale_pred'], llm_predict.long())
                loss = loss_classify
                num_expert = 2
                loss = loss + config.model.rationale_justifiable_evaluator_weight * loss_rationale / num_expert
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss_classify.add(loss_classify.item())
            writer.add_scalar(tag='train_loss', scalar_value=avg_loss_classify.get(), global_step=epoch)
            tqdm.write('==================== validate start ====================')
            val_result, val_avg_loss = self.validate(val_dataloader)
            writer.add_scalars(main_tag='validate_metric', tag_scalar_dict=val_result, global_step=epoch)
            writer.add_scalar(tag='validate_loss', scalar_value=val_avg_loss.get(), global_step=epoch)
            current_val_metric = val_result['accuracy']
            if current_val_metric > best_val_metric:
                best_model_epoch = epoch
                best_val_metric = current_val_metric
                torch.save(self.model.state_dict(), os.path.join(save_path, best_model_name))
        train_end_time = time.time()
        tqdm.write(f"==> training time cost: {train_end_time - train_end_time}s")
        tqdm.write("==================== test start ====================")
        self.model.load_state_dict(torch.load(f=os.path.join(save_path, best_model_name)))
        future_results, label, pred, id, ae, acc = self.test(test_dataloader)
        writer.add_scalars(main_tag="test_result", tag_scalar_dict=future_results)
        return future_results, os.path.join(save_path, best_model_name), best_model_epoch

    def validate(self, dataloader):
        loss_function = BCELoss()
        predict_list = []
        label_list = []
        test_data_iterator = tqdm(dataloader)
        avg_loss_classify = moving_average.RunningAverage()
        self.model.eval()
        for step_n, batch in enumerate(test_data_iterator):
            with torch.no_grad():
                data_cuda = data2gpu.to_gpu(batch_data=batch)
                label = data_cuda['label']
                test_result = self.model(data_cuda)
                loss_classify = loss_function(test_result['classify_pred'], label.float())
                label_list.extend(label.detach().cpu().numpy().tolist())
                predict_list.extend(test_result['classify_pred'].detach().cpu().numpy().tolist())
                avg_loss_classify.add(loss_classify.item())
        return metric.get_metrics(label_list, predict_list), avg_loss_classify

    def test(self, dataloader):
        is_predict_model = config.eval_mode
        if is_predict_model:
            self.model = EFENGModel()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            tqdm.write("==> start predicting, loading model")
            eval_model_path = config.save_param_dir + 'best_' + config.save_model_name
            self.model.load_state_dict(torch.load(eval_model_path))
        pred = []
        label = []
        id = []
        ae = []
        accuracy = []
        predict_data_iterator = tqdm(dataloader)
        self.model.eval()
        for step_n, batch in enumerate(predict_data_iterator):
            with torch.no_grad():
                data_cuda = data2gpu.to_gpu(batch_data=batch)
                batch_label = data_cuda['label']
                predict_result = self.model(data_cuda)
                batch_pred = predict_result['classify_pred']
                cur_labels = batch_label.detach().cpu().numpy().tolist()
                cur_preds = batch_pred.detach().cpu().numpy().tolist()
                label.extend(cur_labels)
                pred.extend(cur_preds)
                ae_list = []
                for index in range(len(cur_labels)):
                    ae_list.append(abs(cur_preds[index] - cur_labels[index]))
                accuracy_list = [1 if ae < 0.5 else 0 for ae in ae_list]
                ae.extend(ae_list)
                accuracy.extend(accuracy_list)
        return metric.get_metrics(label, pred), label, pred, id, ae, accuracy
