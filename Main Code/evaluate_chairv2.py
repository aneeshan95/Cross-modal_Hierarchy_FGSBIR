import pdb
from dataset_chairv2 import *
import time
import torch.nn.functional as F
from matplotlib import pyplot as plt
from Net_Basic_V1 import Net_Basic
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Evaluate_chairv2(model):
    start_time = time.time()
    model.eval()

    parser = argparse.ArgumentParser()
    test_opt = parser.parse_args()

    test_opt.coordinate = '/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/dataset_SBIR/ChairV2_Coordinate'
    test_opt.roor_dir = '/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/dataset_SBIR//chairV2'

    # test_opt.coordinate = '/home/media/GUI_Research/codes_SBIR_CAMP/ChairV2_Coordinate'
    # test_opt.roor_dir = '/home/media/On_the_Fly/Code_ALL/Dataset_Three/ChairV2/'

    test_opt.mode = 'Test'
    test_opt.Train = False
    test_opt.shuffle = False
    test_opt.nThreads = 1
    test_opt.batch_size = 16

    dataset_sketchy_test = CreateDataset_Sketchy(test_opt, on_Fly=False)
    dataloader_sketchy_test = data.DataLoader(dataset_sketchy_test, batch_size=test_opt.batch_size, shuffle=test_opt.shuffle,
                                         num_workers=int(test_opt.nThreads), collate_fn=dataset_sketchy_test.collate_self)

    image_feature_List_ALL = []
    Image_Name = []
    sketch_feature_List_ALL = []
    Sketch_Name = []

    for i_batch, sanpled_batch in enumerate(dataloader_sketchy_test):

        model_input = sanpled_batch['sketch_img'].to(device),   [x.to(device) for x in sanpled_batch['sketch_boxes']], \
                          sanpled_batch['positive_img'].to(device), [x.to(device) for x in sanpled_batch['positive_boxes']]
        sketch_feature_List, image_feature_List = model.Test(*model_input)
        sketch_feature_List_ALL.extend(sketch_feature_List)
        Sketch_Name.extend(sanpled_batch['sketch_path'])


        for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
            if positive_name not in Image_Name:
                Image_Name.append(sanpled_batch['positive_path'][i_num])
                image_feature_List_ALL.append(image_feature_List[i_num])
    rank = torch.Tensor(len(Sketch_Name))

    start_time2 = time.time()
    for num, sketch_matrix in enumerate(sketch_feature_List_ALL):
        distance_list = []
        for count, image_matrix in enumerate(image_feature_List_ALL):
            sketch_F, image_F = model.Hierarchy_Implcit(sketch_matrix, image_matrix, Training=False)
            # print('hamba ', count, num)
            distance_list.append(F.pairwise_distance(sketch_F.unsqueeze(0), image_F.unsqueeze(0)))
        distance_list = torch.stack(distance_list)
        s_name = Sketch_Name[num]
        sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
        position_query = Image_Name.index(sketch_query_name)
        rank[num] = distance_list.le(distance_list[position_query]).sum()

    top1 = rank.le(1).sum().numpy() / rank.shape[0]
    top10 = rank.le(10).sum().numpy() / rank.shape[0]

    print('Test Loop', '-'*50,'\nTime to EValuate:{}'.format((time.time() - start_time)/len(Sketch_Name)))
    print('search time: ',(time.time() - start_time2)/len(Sketch_Name))
    print('Test Accuracy: {:0.5f}/{:0.5f}\n'.format(top1, top10), '-'*50)

    return top1, top10


if __name__ == "__main__":
    model = Net_Basic()
    model.load_state_dict(torch.load('model_Best.pth'))
    print('model loaded')
    model.to('cuda:0')

    top1, top10 = Evaluate_chairv2(model)

    print('Top1_Accuracy: {}, Top10_Accuracy: {}'.format(top1, top10))






