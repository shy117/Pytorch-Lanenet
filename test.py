# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2021/4/24 9:20
"""
import time, os, sys, cv2, warnings
from dataloader import *
from model.model import LaneNet, compute_loss
from average_meter import *
warnings.filterwarnings('ignore')
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cuda'
save_folder = 'seg_result'
os.makedirs(save_folder, exist_ok=True)

if __name__ == '__main__':
	dataset = 'data/training_data_example'
	val_dataset_file = os.path.join(dataset, 'train.txt')
	val_dataset = LaneDataSet(val_dataset_file, stage = 'val')
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
	model = torch.load('checkpoints/012.pth', map_location=DEVICE)
	model.eval()
	for batch_idx, (image_data, binary_label, instance_label) in enumerate(val_loader):
		image_data, binary_label, instance_label = image_data.to(DEVICE),binary_label.type(torch.FloatTensor).to(DEVICE),instance_label.to(DEVICE)
		with torch.set_grad_enabled(False):
			# 预测，并可视化
			net_output = model(image_data)
			seg_logits = net_output["seg_logits"].cpu().numpy()[0]
			# 背景为（0~50）黄色线为（51~200），白色线为（201~255）
			result = (np.argmax(seg_logits, axis=0)*127).astype(np.uint8)       # 此处背景是0，黄色线是127，白色线是254
			cv2.imwrite(os.path.join(save_folder, '{0:04d}.png'.format(batch_idx)), result)
			fig, axs = plt.subplots(1,2)
			axs[0].imshow(image_data.cpu().numpy()[0,0])
			axs[1].imshow(result)
			plt.show()

