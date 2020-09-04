import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from dataset_chairv2 import *
from Net_Basic_V1_4h_2l import Net_Basic, Joint_embdedding
import time
import torch.optim as optim
import torch.utils.data as data
from evaluate_chairv2 import Evaluate_chairv2 as evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main_train(opt):
    model_image = Net_Basic()
    model_sketch = Joint_embdedding()

    model_image.to(device)
    model_image.train()

    model_sketch.to(device)
    model_sketch.train()

    Triplet_Criterion = nn.TripletMarginLoss(margin=0.3).to(device)


    optimizer = optim.Adam(list(model_image.parameters()) + list(model_sketch.parameters()), lr=opt.lr)
    dataset_sketchy_train = CreateDataset_Sketchy(opt)
    dataloader_sketchy_train = data.DataLoader(dataset_sketchy_train, batch_size=opt.batchsize, shuffle=opt.shuffle,
                                               num_workers=int(opt.nThreads),collate_fn=dataset_sketchy_train.collate_self)

    top1_buffer = 0
    top10_buffer = 0
    iter = 0


    for epoch in range(opt.niter):
        for i, sanpled_batch in enumerate(dataloader_sketchy_train, 0):
            model_image.train()
            model_sketch.train()
            iter += 1
            start_time = time.time()
            optimizer.zero_grad()

            sketch_anchor_embedding = model_sketch(sanpled_batch['sketch_img'].to(device),
                                                   sanpled_batch['sketch_coord'].to(device),
                                                   sanpled_batch['seq_len'].to(device))
            #  taking 64 version
            rgb_positive_embedding, _ = model_image(sanpled_batch['positive_img'].to(device))
            rgb_negetive_embedding, _ = model_image(sanpled_batch['negetive_img'].to(device))

            Triplet_Loss = Triplet_Criterion(sketch_anchor_embedding, rgb_positive_embedding, rgb_negetive_embedding)
            loss = Triplet_Loss


            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print('Epoch: {}, Iteration: {}, Time: {}, Total_Loss: {}, Top1: {}, Top10: {}'.format(epoch,
                      i, (time.time() - start_time), Triplet_Loss, top1_buffer, top10_buffer))

            if (i + 0) % opt.save_iter == 0:
                with torch.no_grad():
                    top1, top10 = evaluate(model_sketch, model_image)

                print('Epoch: {}, Iteration: {}, Top1_Accuracy: {}, Top10_Accuracy: {}'.format(epoch, i, top1, top10))

                if top1 > top1_buffer:
                    torch.save(model_sketch.state_dict(), 'model_Best_sketch.pth')
                    torch.save(model_image.state_dict(), 'model_Best_image.pth')
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
    opt.batchsize = 8
    opt.nThreads = 8
    opt.lr = 0.0001
    opt.niter = 200
    opt.save_iter = 100
    opt.load_earlier = False

    print(opt)
    print('Sketchformer Variant')
    main_train(opt)




