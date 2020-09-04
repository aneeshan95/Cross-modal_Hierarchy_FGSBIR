import torch.nn as nn
from dataset_chairv2 import *
from Net_Basic_V1 import Net_Basic#, Resnet_Network
import time
import torch.optim as optim
import torch.utils.data as data
from evaluate_chairv2 import Evaluate_chairv2 as evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def main_train(opt):
    model = Net_Basic()

    model.to(device)
    model.train()

    Triplet_Criterion = nn.TripletMarginLoss(margin=0.3).to(device)


    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    dataset_sketchy_train = CreateDataset_Sketchy(opt)
    dataloader_sketchy_train = data.DataLoader(dataset_sketchy_train, batch_size=opt.batchsize, shuffle=opt.shuffle,
                                               num_workers=int(opt.nThreads))

    top1_buffer = 0
    top10_buffer = 0
    iter = 0

    print('No. of parameters', get_n_params(model))

    with torch.no_grad():
        top1, top10 = evaluate(model)


    for epoch in range(opt.niter):
        for i, sanpled_batch in enumerate(dataloader_sketchy_train, 0):
            model.train()
            iter += 1
            start_time = time.time()
            sketch_anchor_embedding = model(sanpled_batch['sketch_img'].to(device))
            rgb_positive_embedding = model(sanpled_batch['positive_img'].to(device))
            rgb_negetive_embedding = model(sanpled_batch['negetive_img'].to(device))

            Triplet_Loss = Triplet_Criterion(sketch_anchor_embedding, rgb_positive_embedding, rgb_negetive_embedding)
            loss = Triplet_Loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('Epoch: {}, Iteration: {}, Time: {}, Total_Loss: {}, Top1: {}, Top10: {}'.format(epoch,
                            i, (time.time() - start_time), Triplet_Loss, top1_buffer, top10_buffer))

            if (i + 0) % opt.save_iter == 0:
                with torch.no_grad():
                    top1, top10 = evaluate(model)

                print('Epoch: {}, Iteration: {}, Top1_Accuracy: {}, Top10_Accuracy: {}'.format(epoch, i, top1, top10))

                if top1 > top1_buffer:
                    torch.save(model.state_dict(), 'model_Best.pth')
                    top1_buffer, top10_buffer = top1, top10
                    print('Model Updated')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.coordinate = './../../Data/ChairV2_Coordinate'
    opt.roor_dir = './../../Data/ChairV2'
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = True
    opt.batchsize = 12
    opt.nThreads = 8
    opt.lr = 0.0001
    opt.niter = 200
    opt.save_iter = 100
    opt.load_earlier = False
    print(opt)
    main_train(opt)




