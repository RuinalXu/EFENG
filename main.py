import time
from train_model import Trainer
from configurator import config

if __name__ == '__main__':
    print(f"==> load config info: {config}")
    trainer = Trainer()
    result_save_path = './final_result.txt'
    s = time.time()
    metrics, model_save_path, best_epoch = trainer.train()
    e = time.time()
    print(f"cost time = {e - s}")
    print(f"metrics = {metrics}")
    print(f"model save path = {config.save_param_dir}")
    print(f"best metrics train epoch = {best_epoch}")
    with open(result_save_path, 'a') as f:
        f.write(f"cost time = {e - s}\n")
        f.write(f"metrics = {metrics}\n")
        f.write(f"model save path = {config.save_param_dir}\n")
        f.write(f"best metrics train epoch = {best_epoch}\n")
