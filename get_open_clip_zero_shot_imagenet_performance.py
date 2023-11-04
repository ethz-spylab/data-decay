# %%
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import open_clip
from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
IMAGENET_VAL_FOLDER = IMAGENET_FOLDER / "val"
OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS = DATA_FOLDER / "open_clip_zero_shot_imagenet_results"
OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS_VAL = OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS / "val"

IMAGENET1K_COUNT = 1000

# %%

def create_dataloader(image_folder, batch_size=32, shuffle=False, num_workers=4, seed=42):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(image_folder, transform=transform)
    torch.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader
# %%
architecture_dataset = open_clip.list_pretrained()
architecture_dataset
# %%
architecture_dataset[39]
# %%
device = torch.device("cuda:3")
architecture, pretrain_dataset = architecture_dataset[39]
model, _, preprocess = open_clip.create_model_and_transforms(
    architecture, pretrained=pretrain_dataset, device=device)
tokenizer = open_clip.get_tokenizer(architecture)
model.eval()
# %%
import torch
from contextlib import suppress


def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
    


class Args:
    def __init__(self,device):
        self.precision = "amp"
        self.device = device
        self.batch_size = 32
        self.model = "RN50"
args = Args(device)

# %%
def accuracy(output, target, topk=(1,)):
    """Computes top k accuracy
    Args:
        output (torch.Tensor): model output
        target (torch.Tensor): ground truth labels
        topk (tuple): top k values to calculate accuracy for
    Returns:
        list: top k accuracies"""
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

#%%
def get_accuracy_logits_targets(model, classifier, dataloader, args):
    """" Calculates accuracy, logits and targets for a given model, classifier and dataloader
    Args:
        model (torch.nn.Module): model to evaluate
        classifier (torch.nn.Module): classifier to use
        dataloader (torch.utils.data.DataLoader): dataloader to use
        args (Args): arguments
    Returns:
        float: top 1 accuracy
        float: top 5 accuracy
        list: logits
        list: targets
    """
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    tot_targets = []
    tot_logits = []
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
            tot_targets.extend(target)
            tot_logits.append(logits)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5, tot_logits, tot_targets
# %%
imagenet_val_dataloader = create_dataloader(IMAGENET_VAL_FOLDER)
# %%
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
# %%
top1, top5, tot_logits, tot_targets = get_accuracy_logits_targets(
    model, classifier, imagenet_val_dataloader, args)
# %%
topk = (1,5)
tot_logits = torch.cat(tot_logits, dim=0)
tot_preds = tot_logits.topk(max(topk), 1, True, True)[1].t()
tot_targets = [tot_targets[i].reshape(1,1) for i in range(len(tot_targets))]
tot_targets = torch.cat(tot_targets, dim=1)
tot_correct = tot_preds.eq(tot_targets.expand_as(tot_preds))

tot_preds = tot_preds.cpu().numpy()
tot_targets = tot_targets.cpu().numpy()
tot_correct = tot_correct.cpu().numpy()
print("For model:")
for i in topk:
    print(f"Accuracy@{i}: {tot_correct[:i].sum() / len(tot_correct[0]):.4f}")
# %% 
true_positives = np.zeros(IMAGENET1K_COUNT)
false_positives = np.zeros(IMAGENET1K_COUNT)
class_count = np.zeros(IMAGENET1K_COUNT)

for i in range(IMAGENET1K_COUNT):
    true_positives[i] = tot_correct[0, tot_targets[0] == i].sum()
    false_positives[i] = (np.sum(tot_preds[0] == i) - true_positives[i])
    class_count[i]= (np.sum(tot_targets[0] == i))
# %%
precision = true_positives / (true_positives + false_positives)
recall = true_positives / class_count
# %%
def get_precision_recall(preds, targets):
    """ Calculates precision and recall for each class
    Args:
        preds (np.array): predictions
        targets (np.array): targets
    Returns:
        np.array: precision
        np.array: recall"""
    true_positives = np.zeros(IMAGENET1K_COUNT)
    false_positives = np.zeros(IMAGENET1K_COUNT)
    class_count = np.zeros(IMAGENET1K_COUNT)

    corrects = preds == targets

    for i in range(IMAGENET1K_COUNT):
        true_positives[i] = corrects[targets == i].sum()
        false_positives[i] = (np.sum(preds == i) - true_positives[i])
        class_count[i]= (np.sum(targets == i))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / class_count
    return precision, recall
# %%
a,b = get_precision_recall(tot_preds[0], tot_targets[0])
# %%
