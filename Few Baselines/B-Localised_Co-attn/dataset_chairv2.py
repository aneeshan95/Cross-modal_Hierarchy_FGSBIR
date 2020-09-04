import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.transforms.functional as F
import argparse
import pickle
import os
import time
from random import randint
from PIL import Image
import torchvision
from render_sketch_chairv2 import redraw_Quick2RGB
from util_bbox import image_boxes

def get_ransform(opt):
    transform_list = []
    if opt.Train:
        transform_list.extend([transforms.Resize(240), transforms.CenterCrop(224)])
    else:
        transform_list.extend([transforms.Resize(224)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)


class CreateDataset_Sketchy(data.Dataset):
    def __init__(self, opt, on_Fly=False):
        with open(opt.coordinate, 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        with open('./../../Data/ShoeV2_Select_Search_boxes_min10_topN16.pickle', 'rb') as handle:
            self.pic_box_dict = pickle.load(handle)
        self.Skecth_Train_List = [x for x in self.Coordinate if 'train' in x]  # separating trains
        self.Skecth_Test_List = [x for x in self.Coordinate if 'test' in x]    # separating tests
        self.opt = opt
        self.transform = get_ransform(opt)
        self.on_Fly = on_Fly


    def collate_self(self, batch):
        #batch_mod = {}
        if self.opt.Train:
            batch_mod = {'sketch_img': [], 'sketch_boxes': [],
                         'positive_img': [], 'positive_boxes': [],
                         'negetive_img': [], 'negetive_boxes': [],
                         }
            for i_batch in batch:
                batch_mod['sketch_img'].append(i_batch['sketch_img'])
                batch_mod['positive_img'].append(i_batch['positive_img'])
                batch_mod['negetive_img'].append(i_batch['negetive_img'])
                batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
                batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())
                batch_mod['negetive_boxes'].append(torch.tensor(i_batch['negetive_boxes']).float())

            batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
            batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)
            batch_mod['negetive_img'] = torch.stack(batch_mod['negetive_img'], dim=0)


        else:
            batch_mod = {'sketch_img': [], 'sketch_boxes': [], 'sketch_path': [],
                         'positive_img': [], 'positive_boxes': [], 'positive_path': [],
                         }

            for i_batch in batch:
                batch_mod['sketch_img'].append(i_batch['sketch_img'])
                batch_mod['sketch_path'].append(i_batch['sketch_path'])
                batch_mod['positive_img'].append(i_batch['positive_img'])
                batch_mod['positive_path'].append(i_batch['positive_path'])
                batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
                batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())

            batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
            batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)


        return batch_mod

    def __getitem__(self, item):
        sample = {}


        if self.opt.mode == 'Train':
            sketch_path = self.Skecth_Train_List[item]

            positive_sample =  '_'.join(self.Skecth_Train_List[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.opt.roor_dir, 'photo', positive_sample + '.png')


            possible_list = list(range(len(self.Skecth_Train_List)))
            possible_list.remove(item)
            negetive_item = possible_list[randint(0, len(possible_list) - 1)]
            negetive_sample = '_'.join(self.Skecth_Train_List[negetive_item].split('/')[-1].split('_')[:-1])
            negetive_path = os.path.join(self.opt.roor_dir, 'photo', negetive_sample + '.png')

            vector_x = self.Coordinate[sketch_path]
            sketch_img, Sample_len, stroke_boxes = redraw_Quick2RGB(vector_x)

            if self.on_Fly == False:
                sketch_img = Image.fromarray(sketch_img[-1]).convert('RGB')
            else:
                sketch_img = [Image.fromarray(sk_img).convert('RGB') for sk_img in sketch_img]

            positive_img = Image.open(positive_path)
            negetive_img = Image.open(negetive_path)

            n_flip = random.random()
            if n_flip > 0.5:

                if self.on_Fly == False:
                    sketch_img = F.hflip(sketch_img)
                else:
                    sketch_img = [F.hflip(sk_img) for sk_img in sketch_img]

                positive_img = F.hflip(positive_img)
                negetive_img = F.hflip(negetive_img)

            if self.on_Fly == False:
                sketch_img = self.transform(sketch_img)
            else:
                sketch_img = [self.transform(sk_img) for sk_img in sketch_img]

            positive_img = self.transform(positive_img)
            positive_boxes = self.pic_box_dict[positive_sample+'.png']
            negetive_img = self.transform(negetive_img)
            negetive_boxes = self.pic_box_dict[negetive_sample+'.png']

            sample = {'sketch_img': sketch_img, 'sketch_path': self.Skecth_Train_List[item], 'sketch_boxes': stroke_boxes,
                      'positive_img': positive_img, 'positive_path': positive_sample, 'positive_boxes': positive_boxes,
                      'negetive_img': negetive_img, 'negetive_path': negetive_sample, 'negetive_boxes': negetive_boxes,
                      'Sample_Len': Sample_len}


        elif self.opt.mode == 'Test':
            sketch_path = self.Skecth_Test_List[item]

            positive_sample = '_'.join(self.Skecth_Test_List[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.opt.roor_dir, 'photo', positive_sample + '.png')

            vector_x = self.Coordinate[sketch_path]
            # sketch_img, Sample_len = redraw_Quick2RGB(vector_x)    # sain
            sketch_img, Sample_len, stroke_boxes = redraw_Quick2RGB(vector_x)
            if self.on_Fly == False:
                sketch_img = self.transform(Image.fromarray(sketch_img[-1]).convert('RGB'))
            else:
                sketch_img = [self.transform(Image.fromarray(sk_img).convert('RGB')) for sk_img in sketch_img]

            positive_img = self.transform(Image.open(positive_path))
            positive_boxes = self.pic_box_dict[positive_sample + '.png']

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'sketch_boxes': stroke_boxes,
                      'positive_img': positive_img, 'positive_path': positive_sample, 'positive_boxes': positive_boxes,
                      'Sample_Len': Sample_len}

        return sample

    def __len__(self):
        if self.opt.mode == 'Train':
            return len(self.Skecth_Train_List)
        elif self.opt.mode == 'Test':
            return len(self.Skecth_Test_List)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    # opt.coordinate = 'ChairV2_Coordinate'
    # opt.roor_dir = '/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/chairV2/'
    opt.coordinate = 'ShoeV2_Coordinate'
    opt.roor_dir = '/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/dataset_SBIR/shoeV2/'

    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = True
    opt.nThreads = 1
    opt.batchsize = 16
    dataset_sketchy = CreateDataset_Sketchy(opt, on_Fly=True)
    dataloader_sketchy = data.DataLoader(dataset_sketchy, batch_size=opt.batchsize, shuffle=opt.shuffle,
                                         num_workers=int(opt.nThreads))

    for i_batch, sanpled_batch in enumerate(dataloader_sketchy):
        t0 = time.time()
        torchvision.utils.save_image(sanpled_batch['sketch_img'][-1], 'sketch_img.jpg', normalize=True)
        torchvision.utils.save_image(sanpled_batch['positive_img'], 'positive_img.jpg', normalize=True)
        # torchvision.utils.save_image(sanpled_batch['negetive_img'], 'negetive_img.jpg', normalize=True)
        # print(i_batch, sanpled_batch['class_label'], (time.time() - t0))
        #print(len(sanpled_batch['sketch_img']))
        #for i_num in range(len(sanpled_batch['sketch_img'])):
        #    torchvision.utils.save_image(sanpled_batch['sketch_img'][i_num], str(i_num) + 'sketch_img.jpg',
        #                                 normalize=True)