#%%
import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

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
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

#%%

def zero_shot_eval(model, data, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    #logging.info('Starting zero-shot imagenet.')
    print('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    #logging.info('Building zero-shot classifier')
    print('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    #logging.info('Using classifier')
    print('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results

# %%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"
RUN10 = Path("/data/cc3m/carlini_logs/logs/run_10/2023_03_31-09_33_50-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
RUN11 = Path("/data/cc3m/carlini_logs/logs/run_11/2023_03_31-09_33_50-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
IMAGENET_TEST_FOLDER = IMAGENET_FOLDER / "val"
IMAGENET_FOLDERS_CLASSES_MAPPING = DATA_FOLDER / "imagenet_folder_label_mapping.txt"

#%%
import torch
from torchvision import datasets, transforms

def create_dataloader(image_folder, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(image_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

# Create dataloaders for ImageNet train and test datasets
imagenet_train_dataloader = create_dataloader(IMAGENET_TRAIN_FOLDER)
imagenet_test_dataloader = create_dataloader(IMAGENET_TEST_FOLDER)

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
    'imagenet-val': DataloaderWrapper(imagenet_test_dataloader),
    'imagenet-v2': DataloaderWrapper(imagenet_train_dataloader),
}

#%%
import open_clip
device = torch.device("cuda")

model_10, _, preprocess_10 = open_clip.create_model_and_transforms(
    'RN50', 
    pretrained=str(RUN10), device=device)


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
#%%
results_10 = zero_shot_eval(model_10, data, args)

# %%


model_11, _, preprocess_11 = open_clip.create_model_and_transforms(
    'RN50', 
    pretrained=str(RUN11), device=device)

results_11 = zero_shot_eval(model_11, data, epoch, args, tokenizer)