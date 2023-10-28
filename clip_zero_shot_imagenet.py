#%%
import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from precision import get_autocast

import torch
import torch.nn.functional as F

import numpy as np

from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"
RUN10 = Path("/data/cc3m/carlini_logs/logs/run_10/2023_03_31-09_33_50-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
RUN11 = Path("/data/cc3m/carlini_logs/logs/run_11/2023_03_31-09_33_50-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
IMAGENET_VAL_FOLDER = IMAGENET_FOLDER / "val"
IMAGENET_FOLDERS_CLASSES_MAPPING = DATA_FOLDER / "imagenet_folder_label_mapping.txt"

# %%
import torch
from torchvision import datasets, transforms

def create_dataloader(image_folder, batch_size=32, shuffle=False, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(image_folder, transform=transform)
    torch.manual_seed(42)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

# Create dataloaders for ImageNet train and test datasets
imagenet_train_dataloader = create_dataloader(IMAGENET_TRAIN_FOLDER)
imagenet_val_dataloader = create_dataloader(IMAGENET_VAL_FOLDER)

#%%
# make data dict s.t. data['imagenet-val'].dataloader is the val dataloader
#data = {
#    'imagenet-val': imagenet_test_dataloader,
#    'imagenet-v2': imagenet_train_dataloader,
#}
class DataloaderWrapper:
    def __init__(self, dataloader):
        self.dataloader = dataloader
data = {
    'imagenet-val': DataloaderWrapper(imagenet_val_dataloader),
}
#%%
import open_clip
device = torch.device("cuda:3")

model_10, _, preprocess_10 = open_clip.create_model_and_transforms(
    'RN50', 
    pretrained=str(RUN10), device=device)

model_10.eval()
#%%
# fill in the args
class Args:
    def __init__(self):
        self.precision = "amp"
        self.device = device
        self.distributed = False
        self.horovod = False
        self.batch_size = 32
        self.model = "RN50"
args = Args()

tokenizer = open_clip.get_tokenizer('RN50')
# %%
print('Building zero-shot classifier')
autocast = get_autocast(args.precision)
with autocast():
    classifier = build_zero_shot_classifier(
        model_10,
        tokenizer=tokenizer,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=args.device,
        use_tqdm=True,
    )

# %%

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk], pred

#%%
def run2(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    tot_targets = []
    tot_preds = []
    print("Running...")
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            [acc1, acc5], pred = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            tot_targets.extend(target)
            tot_preds.append(pred)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5, tot_targets, tot_preds
# %%
top1, top5, tot_targets, tot_preds = run2(model_10, classifier, data['imagenet-val'].dataloader, args)
# %%
# top 1 accuracy
tot_targets2 = np.array([tot_targets[i].cpu().numpy() for i in range(len(tot_targets))])
tot_preds2 = ([tot_preds[i][0].cpu().numpy() for i in range(len(tot_preds))])
tot_preds2 = np.concatenate( tot_preds2, axis=0 )
# %%
np.sum(tot_targets2 == tot_preds2) / len(tot_targets2)

# %%
tot_preds3 = [tot_preds[i].T for i in range(len(tot_preds))]
tot_preds3 = torch.cat(tot_preds3, dim=0)
tot_targets3 = [tot_targets[i].reshape(1,1) for i in range(len(tot_targets))]
tot_targets3 = torch.cat(tot_targets3, dim=0)
correct = tot_preds3.eq(tot_targets3.expand_as(tot_preds3))
# %%
top1 = correct[:,:5].sum() / len(correct)
# %%
tot_preds3 = [tot_preds[i] for i in range(len(tot_preds))]
tot_preds3 = torch.cat(tot_preds3, dim=1)
tot_targets3 = [tot_targets[i].reshape(1,1) for i in range(len(tot_targets))]
tot_targets3 = torch.cat(tot_targets3, dim=1)
correct = tot_preds3.eq(tot_targets3.expand_as(tot_preds3))
# %%
top1 = correct[:1].sum() / len(correct[0])
# %%













# %%
tot_preds3 = ([tot_preds[i].cpu().numpy().T for i in range(len(tot_preds))])
tot_preds3 = np.concatenate( tot_preds3, axis=0 )
tot_targets3 = np.tile(tot_targets2, tot_preds3.shape)
np.sum(tot_targets3 == tot_preds3) / len(tot_targets3)



# %%
#tot_preds3 = ([np.concatenate(tot_preds[i].cpu().numpy().T, axis=0) for i in range(len(tot_preds))])
tot_preds3 = ([tot_preds[i].cpu().numpy().T for i in range(len(tot_preds))])
# %%
tot_preds3
# %%
len(tot_preds3[0])
# %%
tot_preds3[0]
# %%
tot_preds3 = np.concatenate( tot_preds3, axis=0 )
# %%
tot_targets[0]
# %%
tot_preds3[1].T.shape
# %%
len(tot_preds3)
# %%
tot_targets2
# %%
tot_targets3 = np.tile(tot_targets2, tot_preds3.shape)
# %%
print("hi")
# %%
