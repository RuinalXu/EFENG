import argparse
from easydict import EasyDict

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--data_path', type=str, default='./datasets/')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--bert_path', type=str, default='/path/to/bert-base-uncased')
parser.add_argument('--save_param_dir', type=str, default='./param_model')
parser.add_argument('--save_tensorboard_dir', type=str, default='./logs/tensorlog')
parser.add_argument('--save_model_name', type=str, default='efeng')
parser.add_argument('--eval_mode', type=bool, default=False)
parser.add_argument('--rationale_justifiable_evaluator_weight', type=float, default=1)

args = parser.parse_args()

config_dict = {
    'epochs': args.epochs,
    'max_len': args.max_len,
    'data_path': args.data_path,
    'batch_size': args.batch_size,

    'emb_dim': args.emb_dim,
    'optimizer': {
        'lr': args.lr,
        'weight_decay': args.weight_decay
    },
    'bert_path': args.bert_path,
    'save_model_name': args.save_model_name,
    'save_param_dir': args.save_param_dir,
    'save_tensorboard_dir': args.save_tensorboard_dir,
    'model': {
        'mlp': {
            'dims': [384],
            'dropout': 0.2
        },
        'rationale_justifiable_evaluator_weight': args.rationale_justifiable_evaluator_weight
    },
    'eval_mode': args.eval_mode,
    'use_cuda': False,
    'aggregator_dim': 3072
}

config = EasyDict(config_dict)