from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import os
from ssr_kernel import *
from utils import TrainSetLoader
from torchvision import transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='/root/data/test')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='SSR_4xSR')
    return parser.parse_args()


def test(cfg, loadname, net, mode='test'):
    psnr_list = []
    psnr_list_r = []
    psnr_list_m1 = []
    psnr_list_r_m1 = []
    psnr_list_m2 = []
    psnr_list_r_m2 = []
    psnr_list_m3 = []
    psnr_list_r_m3 = []

    lr_path = os.path.abspath(cfg.testset_dir + '/LR_x4')
    file_len = int(len(os.listdir(lr_path)) / 2)
    for idx in range(file_len):
        name_id = '%.4d' % (idx + 1)
        LR_left  = Image.open(lr_path + f'/{name_id}_L.png')
        LR_right = Image.open(lr_path + f'/{name_id}_R.png')

        LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
        LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
        LR_left, LR_right = Variable(LR_left).cuda(), Variable(LR_right).cuda()

        with torch.no_grad():
            output_list = net(LR_left, LR_right)
            SR_left0, SR_right0 = output_list[:2]
            SR_left1, SR_right1, SR_left2, SR_right2 = output_list[2:6]
            SR_left3, SR_right3, SR_left4, SR_right4 = output_list[6:10]
            print(f'{name_id} {torch.cuda.max_memory_allocated() / 1e9:.3f} G')
            SR_left0, SR_right0 = torch.clamp(SR_left0, 0, 1), torch.clamp(SR_right0, 0, 1)
            SR_left1, SR_right1 = torch.clamp(SR_left1, 0, 1), torch.clamp(SR_right1, 0, 1)
            SR_left2, SR_right2 = torch.clamp(SR_left2, 0, 1), torch.clamp(SR_right2, 0, 1)
            SR_left3, SR_right3 = torch.clamp(SR_left3, 0, 1), torch.clamp(SR_right3, 0, 1)
            SR_left4, SR_right4 = torch.clamp(SR_left4, 0, 1), torch.clamp(SR_right4, 0, 1)
            torch.cuda.empty_cache()

        if 0:
            for i in [0,1,2,3,4]:
                # save all the intermediate pictures
                left = [SR_left0, SR_left1, SR_left2, SR_left3, SR_left4]
                right = [SR_right0, SR_right1, SR_right2, SR_right3, SR_right4]
                save_path = f'./results/ssr/{i}'
                os.makedirs(save_path, exist_ok=True)
                SR_left_img = transforms.ToPILImage()(torch.squeeze(left[i].data.cpu(), 0))
                SR_left_img.save(save_path + '/' + name_id + '_L.png')
                SR_right_img = transforms.ToPILImage()(torch.squeeze(right[i].data.cpu(), 0))
                SR_right_img.save(save_path + '/' + name_id + '_R.png')

        if 1:
            # only save the last output picture
            save_path = './results/' + cfg.model_name + f'/{mode}/'
            os.makedirs(save_path, exist_ok=True)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left4.data.cpu(), 0))
            SR_left_img.save(save_path + '/' + name_id + '_L.png')
            SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right4.data.cpu(), 0))
            SR_right_img.save(save_path + '/' + name_id + '_R.png')


if __name__ == '__main__':
    cfg = parse_args()
    net = SSR().cuda()
    net = torch.nn.DataParallel(net)
    print(f'{torch.cuda.max_memory_allocated() / 1e9:.3f} G0')
    net.eval()
    num_epoch = 57
    loadname = f'./checkpoints/SSR_4xSR_epoch{num_epoch}.pth.tar'
    print(loadname)
    model = torch.load(loadname)
    net.load_state_dict(model['state_dict'])

    test(cfg, loadname, net)

    print('Finished!')
