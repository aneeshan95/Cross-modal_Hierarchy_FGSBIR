
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
    # test_opt.coordinate = '/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/dataset_SBIR/ChairV2_Coordinate'
    # test_opt.roor_dir = '/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/dataset_SBIR/chairV2'
    test_opt.coordinate = '/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/dataset_SBIR/ShoeV2_Coordinate'
    test_opt.roor_dir = '/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/dataset_SBIR/ShoeV2'
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
        for image_matrix in image_feature_List_ALL:
            sketch_F, image_F = model.CAMP_Interact(sketch_matrix, image_matrix)
            distance_list.append(F.pairwise_distance(sketch_F.unsqueeze(0), image_F.unsqueeze(0)))

        s_name = Sketch_Name[num]
        sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
        position_query = Image_Name.index(sketch_query_name)

        # print(s_name)
        # 3775693_v1_Chestnut_1
        #   '/test/CHARUF012YEL-UK_v1_MustardYellow_1' \
        # 9910-02-carbon
        # CHABRA003GRY-UK_v1_PearlGrey_1
        # testname = '/test/SOFWLS045BLU-UK_v1_NavyVelvet_3'
        # testname = '/test/SOFMLI002GRY-UK_v1_Gra_2'
        # if s_name == testname:
        #     print(s_name, sketch_query_name,position_query)
        #     alpha = sorted(distance_list)
        #     new_im = Image.new('RGB', (258 * 6, 258*6))
        #     for k in range(30):
        #         photo_img = Image.open(test_opt.roor_dir + '/photo/'
        #                                + Image_Name[distance_list.index(alpha[k])] + '.png', 'r').resize((256, 256))
        #         # photo_img.save('/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/dataset_SBIR/chairV2/' + Image_Name[distance_list.index(alpha[k])] + '.png')
        #         new_im.paste(photo_img, ((k%6)*258, (k//6)*258))
        #     photo_img = Image.open(test_opt.roor_dir + '/photo/'
        #                            + sketch_query_name + '.png', 'r').resize((256, 256))
        #     sktch_img = Image.open(test_opt.roor_dir + '/sketch/'
        #                            + s_name[s_name.rfind('/') + 1:] + '.png', 'r').resize((256, 256))
        #     new_im.paste(photo_img, (2 * 258, 5 * 258))
        #     new_im.paste(sktch_img, (3 * 258, 5 * 258))
        #     new_im.show()
        #     new_im.save('/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/dataset_SBIR/chairV2/' + s_name[s_name.rfind('/') + 1:] + '.png')
        #     print('Success')

        distance_list = torch.stack(distance_list)
        rank[num] = distance_list.le(distance_list[position_query]).sum()

    top1 = rank.le(1).sum().numpy() / rank.shape[0]
    top10 = rank.le(10).sum().numpy() / rank.shape[0]

    print('Time to EValuate:{}'.format((time.time() - start_time)/len(Sketch_Name)))
    print('Time to EValuate:{}'.format((time.time() - start_time2)/len(Sketch_Name)))

    return top1, top10


if __name__ == "__main__":
    model = Net_Basic()
    model.load_state_dict(torch.load('/vol/research/sketchCV/SWIRE_Project/Successive_Exp/SBIR/debug/Final_Ayan_Camp/model_Best.pth'))
    print('model loaded')
    model.to('cuda:0')

    with torch.no_grad():
        top1, top10 = Evaluate_chairv2(model)

    print('Top1_Accuracy: {}, Top10_Accuracy: {}'.format(top1, top10))






