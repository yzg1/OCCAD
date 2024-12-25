from __future__ import print_function
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from dataset import * 
from model import * 
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='./data', help='data root path')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--log', type=str, default='log', help='log path')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-6, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
parser.add_argument('--start_epoch', type=int, default=1, help='start epoch number for resume')
opt = parser.parse_args()

# GPU和随机种子设置
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 数据加载器设置
batch_size_val = 4
dataset = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                   pc_occ_root=os.path.join(opt.dataRoot, 'pc_occ'),
                   filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                   mode='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_val = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                       pc_occ_root=os.path.join(opt.dataRoot, 'pc_occ'),
                       filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                       mode='val')
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size_val,
                                             shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset)
len_dataset_val = len(dataset_val)
print('training set num', len_dataset)
print('validation set num', len_dataset_val)
print("Dataset total unique data paths:", len(set(dataset.data_paths)))
print("Dataset total raw data paths:", len(dataset.data_paths))

# Print the unique sample identifiers for debugging
sample_ids = []
for i, data in enumerate(dataloader):
    _, _, name, view_id = data
    sample_ids.extend(list(zip(name, view_id)))

print(f"Unique training samples: {len(set(sample_ids))}")
print(f"Total training samples: {len(sample_ids)}")

cudnn.benchmark = True

# 创建路径
model_path = opt.model
log_path = opt.log
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
logger = SummaryWriter(log_path)

# 创建网络
network = OccupancyNetwork()
network = torch.nn.DataParallel(network)
network.cuda()

checkpoint_path = os.path.join(model_path, 'checkpoint.pt')
model_path = os.path.join(model_path, 'occnet_best.pt')

# 初始化训练状态
it_step = 0
min_loss = 1e8
best_epoch = 0
start_epoch = opt.start_epoch

# 加载检查点
if opt.resume and os.path.exists(checkpoint_path):
    try:
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        it_step = checkpoint['it_step']
        min_loss = checkpoint['min_loss']
        best_epoch = checkpoint['best_epoch']
        print(f'Resuming from epoch {start_epoch}, iteration {it_step}')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        start_epoch = 1  # If loading fails, start from scratch
elif os.path.exists(model_path):
    network.load_state_dict(torch.load(model_path))
    print('Loaded pre-trained occnet weights.')
else:
    print('No pre-trained weights found. Training from scratch.')

# Create Loss Module and optimizer
criterion_bce = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(network.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

# Create TensorBoard writer
logger = SummaryWriter(log_path)

# img_transform_train = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
# ])

# Training
loss_train = 0
loss_n1 = 0
it_step = 0

# 用于记录每 10 步的训练损失（逐步记录曲线）
train_loss_step = []

for epoch in range(start_epoch, opt.nepoch + 1):
    network.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{opt.nepoch}", ncols=100)
    epoch_loss = 0
    epoch_n = 0  # 每个 epoch 的步数计数

    for it, data in enumerate(pbar, 0):
        it_step += 1
        epoch_n += 1

        optimizer.zero_grad()
        img, occ_data, _, _ = data
        img = img.cuda()
        occ_p = occ_data[None].cuda()
        occ_val = occ_data['occ'].cuda()

        occ_pre, _ = network(img, occ_p)
        loss = criterion_bce(occ_pre, occ_val)

        loss.backward()
        
        # 训练中梯度爆炸会导致不稳定，引入梯度裁剪
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 记录当前步的损失
        loss_train += loss.item()
        loss_n1 += 1
        epoch_loss += loss.item()

        # 每 10 步记录一次逐步损失
        if it_step % 10 == 0:
            train_loss_step.append(loss.item())  # 保存逐步损失
            logger.add_scalar('train/loss_step', loss.item(), it_step)
            
        torch.cuda.empty_cache()    
        
        # 更新进度条显示当前损失
        pbar.set_postfix(loss=loss.item())

    # 计算并记录当前 epoch 的平均损失
    loss_train /= loss_n1  # 总平均损失
    epoch_avg_loss = epoch_loss / epoch_n  # 当前 epoch 的平均损失
    logger.add_scalar('train/loss_avg', epoch_avg_loss, epoch)

    # 打印并记录到日志
    print(f"Epoch {epoch}: average loss = {epoch_avg_loss:.6f}")
    pbar.set_postfix(avg_loss=epoch_avg_loss)

    # Validation
    loss_val = 0
    loss_n = 0
    network.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader_val, desc=f"Epoch {epoch}/{opt.nepoch} Validation", ncols=100)
        for it, data in enumerate(pbar, 0):
            img, occ_data, _, _ = data
            img = img.cuda()
            occ_p = occ_data[None].cuda()
            occ_val = occ_data['occ'].cuda()

            occ_pre, _ = network(img, occ_p)
            loss = criterion_bce(occ_pre, occ_val)

            loss_val += loss.item()
            loss_n += 1
            
            # 删除无用变量并清空显存
            del img, occ_data, occ_p, occ_val, occ_pre
            torch.cuda.empty_cache()
            # 更新进度条显示
            pbar.set_postfix(loss=loss.item())

        loss_val /= loss_n
        logger.add_scalar('val/loss', loss_val, it_step)

    # 保存最佳模型
    if loss_val < min_loss:
        min_loss = loss_val
        best_epoch = epoch
        torch.save(network.state_dict(), model_path)

    # 保存检查点
    checkpoint = {
        'epoch': epoch,
        'it_step': it_step,
        'model_state_dict': network.state_dict(),
        'min_loss': min_loss,
        'best_epoch': best_epoch
    }
    torch.save(checkpoint, checkpoint_path)

print('Training done!')
print('Best epoch is', best_epoch)


