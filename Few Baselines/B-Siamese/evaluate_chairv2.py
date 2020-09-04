
from dataset_chairv2 import *
import time
import torch.nn.functional as F
from matplotlib import pyplot as plt
from Net_Basic_V1 import Net_Basic#, Resnet_Network
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Evaluate_chairv2(model):
    start_time = time.time()
    model.eval()

    parser = argparse.ArgumentParser()
    test_opt = parser.parse_args()
    test_opt.coordinate = 'ChairV2_Coordinate'
    test_opt.roor_dir = '/vol/research/sketchCV/datasets/ChairV2'
    test_opt.mode = 'Test'
    test_opt.Train = False
    test_opt.shuffle = False
    test_opt.nThreads = 1
    test_opt.batch_size = 16

    dataset_sketchy_test = CreateDataset_Sketchy(test_opt, on_Fly=False)
    dataloader_sketchy_test = data.DataLoader(dataset_sketchy_test, batch_size=test_opt.batch_size, shuffle=test_opt.shuffle,
                                         num_workers=int(test_opt.nThreads))

    Image_Array = torch.FloatTensor().to(device)
    Image_Name = []
    Sketch_Array = torch.FloatTensor().to(device)
    Sketch_Name = []

    for i_batch, sanpled_batch in enumerate(dataloader_sketchy_test):
        #print(i_batch)
        sketch_feature = model(sanpled_batch['sketch_img'].to(device))
        Sketch_Array = torch.cat((Sketch_Array, sketch_feature.detach()))
        Sketch_Name.extend(sanpled_batch['sketch_path'])

        batch_check = []
        for num, x in enumerate(sanpled_batch['positive_path']):
            if x not in Image_Name:
                batch_check.append(num)
                Image_Name.append(x)

        batch_image = sanpled_batch['positive_img'][batch_check, :, :, :]

        if batch_image.shape[0] > 0:
            image_feature = model(batch_image.to(device))
            Image_Array = torch.cat((Image_Array, image_feature.detach()))

    rank = torch.Tensor(Sketch_Array.shape[0])
    start_time2 = time.time()
    for ik in range(Sketch_Array.shape[0]):
        sketch_query_name = '_'.join(Sketch_Name[ik].split('/')[-1].split('_')[:-1])
        position_query = Image_Name.index(sketch_query_name)
        target_distance = F.pairwise_distance(Sketch_Array[ik, :].unsqueeze(0),
                                              Image_Array[position_query].unsqueeze(0))
        distance = F.pairwise_distance(Sketch_Array[ik, :], Image_Array)
        rank[ik] = distance.le(target_distance).sum()

    top1 = rank.le(1).sum().numpy() / rank.shape[0]
    top10 = rank.le(10).sum().numpy() / rank.shape[0]

    print('Time to EValuate:{}'.format(time.time() - start_time))
    print('Time to EValuate:{}'.format((time.time() - start_time2)/Sketch_Array.shape[0]))

    return top1, top10


if __name__ == "__main__":
    # model = Net_Basic()
    model = Resnet_Network()
    model.load_state_dict(torch.load('model_Best.pth'))
    print('model loaded')
    model.to('cuda:0')

    with torch.no_grad():
        top1, top10 = Evaluate_chairv2(model)

    print('Top1_Accuracy: {}, Top10_Accuracy: {}'.format(top1, top10))






