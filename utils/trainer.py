import time
from loops import train_step, valid

class Trainer():
    def __init__(self, train_config, val_config, logger) -> None:
        self.num_epoch = train_config['num_epoch']
        self.train_config = train_config
        self.val_config = val_config
        self.logger = logger

    def _train_step(self, model, batch, criterion, device):
        return train_step(model, batch, criterion, device, self.train_config)

    def _validation_steps(self, dataloader, model, device):
        return valid(dataloader, model, device, self.val_config)
    
    def run(self, train_dataloader, valid_dataloader, model, optimizer, criterion, start_epoch, utiler, device):
        for epoch in range(start_epoch, self.num_epoch):
            self._validation(valid_dataloader, model, device)
            self._train_one_epoch(train_dataloader, model, optimizer, criterion, epoch, device)
            utiler.save_ckp(model, optimizer, epoch)

    def _train_one_epoch(self, dataloader, model, optimizer, criterion, cur_epoch, device):
        print('\nTraining...')

        model.train()
        model.to(device)
        start = time.time()
        loss_lst = []
        for batch_idx, batch in enumerate(dataloader):
            l_r = self._adjust_l_r(optimizer, cur_epoch, batch_idx, len(dataloader))
            loss = self._train_step(model, batch, criterion, device)

            optimizer.zero_grad()
            loss['loss'].backward()
            optimizer.step()

            loss = {k:v for k, v in loss.items()}
            loss['loss'] = loss['loss'].item()
            loss_lst.append(loss)
            

            now = time.time()
            time_used = int(now - start)
            if self.train_config['DEBUG'] == False:
                if batch_idx % (len(dataloader) // 10) == 0:
                    self.logger.log(f'epoch:[{cur_epoch:>3d}/{self.num_epoch:>3d}], batch:[{batch_idx:>3d}/{len(dataloader):>3d}], l_r: {l_r:>6f}, time used: {(time_used // 60):>2d} min {(time_used % 60):>2d} sec, loss:[{loss}]')
            else:
                self.logger.log(f'epoch:[{cur_epoch:>3d}/{self.num_epoch:>3d}], batch:[{batch_idx:>3d}/{len(dataloader):>3d}], l_r: {l_r:>6f}, time used: {(time_used // 60):>2d} min {(time_used % 60):>2d} sec, loss:[{loss}]')
            
            if self.train_config['DEBUG'] == True:
                if batch_idx == 5:
                    return []


        end = time.time()
        time_used = int(end - start)

        self.logger.log(f'\nepoch:[{cur_epoch:>3d} done!, time to run training for this epoch: {(time_used // 60):>2d} min {(time_used % 60):>2d} sec')
        return loss_lst

    def _validation(self, dataloader, model, device):
        print('\nValidation...')

        model.eval()
        model.to(device)
        start = time.time()
        score = self._validation_steps(dataloader, model, device)
        end = time.time()
        time_used = int(end - start)
        self.logger.log(f'score: {score}, time to run validation: {(time_used // 60):>2d} min {(time_used % 60):>2d} sec.')

    def _adjust_l_r(self, optimizer, cur_epoch, batch_idx, len_data_loader):
        schedule = self.train_config['schedule']
        if isinstance(schedule, str):
            assert schedule == 'polylr', 'Error: schedule should be polylr!'
            cur_iter = cur_epoch * len_data_loader + batch_idx
            max_iter_num = self.num_epoch * len_data_loader
            l_r = self.train_config['l_r'] * (1 - float(cur_iter) / max_iter_num) ** 0.9
        elif isinstance(schedule, tuple):
            l_r = self.train_config['l_r']
            for i in range(len(schedule)):
                if cur_epoch < schedule[i]:
                    break
                l_r = l_r * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = l_r
        return l_r
