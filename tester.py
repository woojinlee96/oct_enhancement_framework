import torch.utils.data
import torch.nn as nn
import torchvision

from loss import GeneratorLoss
from model import Generator, Discriminator
import utils
import data_loader

import warnings
import os
import time
from scipy import io

def warn_func():
    warn_message = 'warn_func() is deprecated, use new_function() instead'
    warnings.warn(warn_message)

warnings.filterwarnings(action='ignore')
warn_func()

model_name = '211021'

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

'''Dataset setting'''
path = '/home/leewj/projects/projects/oct_enhancement/k_space/dataset'

trainloader = data_loader.get_loader(dataset_path=path, phase='train', shuffle=True, batch_size=1, num_workers=1)
valloader = data_loader.get_loader(dataset_path=path, phase='val', shuffle=True, batch_size=1, num_workers=1)
testloader = data_loader.get_loader(dataset_path=path, phase='test', shuffle=True, batch_size=1, num_workers=1)

netG = Generator()

generator_criterion = GeneratorLoss()

netG = nn.DataParallel(netG, device_ids=[2], output_device=2)

netG.to(device)
generator_criterion.to(device)
netG.to(device)
Gcheckpoint = torch.load('output/netG_%s.pth' % (model_name))
netG.load_state_dict(Gcheckpoint['State_dict'])

results = {'d_loss': [], 'g_loss': [],'d_score': [], 'g_score': []}

TRIAL = 5
saving = False
vis = True

for trial in range(1, TRIAL + 1):
    netG.eval()
    with torch.no_grad():
        data_stft, data, target = next(iter(testloader))

        data_stft = data_stft.squeeze(dim=0)
        data = data.squeeze(dim=0).to(device)
        target = target.squeeze(dim=0).to(device)

        data_stft1 = data_stft[0:128].to(device)
        data1 = data[0:128].to(device)
        out1 = netG(data_stft1, data1)

        data_stft2 = data_stft[128:256].to(device)
        data2 = data[128:256].to(device)
        out2 = netG(data_stft2, data2)

        data_stft3 = data_stft[256:384].to(device)
        data3 = data[256:384].to(device)
        out3 = netG(data_stft3, data3)

        data_stft4 = data_stft[384:512].to(device)
        data4 = data[384:512].to(device)
        out4 = netG(data_stft4, data4)

        out = torch.cat([out1, out2, out3, out4], dim=0)
        out = torch.reshape(out, [1024,2048])

        input_img = torch.reshape(data, [1024,2048]).permute(1,0)
        target_img = torch.reshape(target, [1024,2048]).squeeze().permute(1,0)
        out = out.permute(1,0)

        if vis:
            result_show = torchvision.utils.make_grid((torch.stack((input_img.unsqueeze(dim=0), target_img.unsqueeze(dim=0), out.unsqueeze(dim=0)), dim=0)))
            utils.imshow(result_show.cpu().squeeze(), title='')

        if saving:
            index = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
            out_path = 'output_image/' + model_name + '/%s'%(index)

            if not os.path.exists(out_path):
                os.makedirs(out_path)
            result = {
                'input' : input_img.squeeze().detach().cpu().numpy()*25+105,
                'real' : target_img.squeeze().detach().cpu().numpy()*25+105,
                'fake' : out.squeeze().detach().cpu().numpy()*25+105,
            }
            io.savemat(out_path + '/result.mat', result)

