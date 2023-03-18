from models.model import PAN
from utils.trainer import Trainer
from utils.utilers import Utiler
import os
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch import optim
from loss.pan_loss import PANLoss
from datasets.data import PAN_CTW
from utils.utilers import get_config

def main():
    config = get_config('./pan-config.yaml')
    utiler = Utiler(config['train_cfg']['model_save_path'], config['train_cfg']['ckp_path'])
    utiler.apply()

    device = config['train_cfg']['device']
    model = PAN(config['backbone'], config['neck_param'], config['head_param'])
    print(model)

    criterion = PANLoss(config['loss_text_param'], config['loss_kernel_param'], config['loss_emb_params'])

    optimizer = optim.Adam(model.parameters(), lr=config['train_cfg']['l_r'])
    print(config['train_data_cfg'])
    train_dataset = PAN_CTW(**config['train_data_cfg'])
    train_dataloader = DataLoader(train_dataset, config['train_cfg']['b_s'], num_workers=2, shuffle=True, drop_last=True)

    val_dataset = PAN_CTW('test')
    val_dataloader = DataLoader(val_dataset, 1, shuffle=False)

    if os.path.exists(config['train_cfg']['model_save_path']):
        start_epoch = utiler.load_ckp(model, optimizer, device)
    else:
        start_epoch = 0
        model.to(device)
        print(f'No ckp found\ntraining will start from srcatch')

    trainer = Trainer(config['train_cfg'])
    trainer.run(train_dataloader,
                val_dataloader,
                model,
                optimizer,
                criterion,
                start_epoch,
                utiler,
                device)

if __name__ == "__main__":
    main()






