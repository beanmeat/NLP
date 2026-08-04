from torch.utils.tensorboard import SummaryWriter

write = SummaryWriter(log_dir='./test_logs')

for step in range(100):
    write.add_scalar('scaler/y=x', step, step)
    write.add_scalar('scaler/y=x^2', step ** 2, step)
write.close()