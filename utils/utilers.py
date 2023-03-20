import pathlib
import numpy as np
import torch
import os
import yaml


class Utiler:
    def __init__(self, model_save_path, ckp_path, logger, seed=2022) -> None:
        self.ckp_path = ckp_path
        self.model_save_path = model_save_path
        self.seed = seed
        self.logger = logger
    
    def apply(self):
        self._mkdir()
        self._set_seed()

    def load_ckp(self, model, optimizer, device):
        checkpoint = torch.load(self.model_save_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        self.logger.log(f'Loading ckp from {self.ckp_path}\ntraining will start from epoch {start_epoch}')
        return start_epoch

    def save_ckp(self, model, optimizer, epoch):
        torch.save({'optimizer': optimizer.state_dict(), 
            'model': model.state_dict(), 
            'epoch': epoch + 1}, 
            self.model_save_path)
        self.logger.log(f'Ckp...\nmodel will be saved to {self.model_save_path}')

    def _mkdir(self):
        pathlib.Path(self.ckp_path).mkdir(parents=True, exist_ok=True)
        self.logger.log(f'Making dir for ckp\nckp path is {self.ckp_path}')

    def _set_seed(self):
        """
        set random seed
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.logger.log(f'Setting seed to {self.seed}')

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, max_len=-1):
        self.val = []
        self.count = []
        self.max_len = max_len
        self.avg = 0

    def update(self, val, n=1):
        self.val.append(val * n)
        self.count.append(n)
        if self.max_len > 0 and len(self.val) > self.max_len:
            self.val = self.val[-self.max_len:]
            self.count = self.count[-self.max_len:]
        self.avg = sum(self.val) / sum(self.count)


def write_result_ctw(image_name, outputs, result_path):
    bboxes = outputs['bboxes']

    lines = []
    for i, bbox in enumerate(bboxes):
        bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
        values = [int(v) for v in bbox]
        line = '%d' % values[0]
        for v_id in range(1, len(values)):
            line += ',%d' % values[v_id]
        line += '\n'
        lines.append(line)

    file_name = '%s.txt' % image_name
    file_path = os.path.join(result_path, file_name)
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line)

def get_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config
