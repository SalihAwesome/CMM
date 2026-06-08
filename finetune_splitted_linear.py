import os
import time
import torch
import copy
import wandb
import random
import numpy as np
from torchinfo import summary

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.cl_utils import get_dataset_and_classifier_for_split
from src.merging.task_vectors import TaskVector
from src.linearize import LinearizedImageEncoder
from src.eval import evaluate,  eval_single_dataset
# Import your new utility functions
from src.utils.merge_utils import compute_adaptive_alpha, log_adaptive_alpha

def interpolate_weights(theta_0, theta_1, alpha, fisher_mat):
    assert len(fisher_mat) == 2
    theta_0 = {k:v.cpu() for k,v in theta_0.items() if isinstance(v, torch.Tensor)}
    theta_1 = {k:v.cpu() for k,v in theta_1.items() if isinstance(v, torch.Tensor)}
    fisher_mat = [{k:v.cpu() for k,v in f.items() if isinstance(v, torch.Tensor) } for f in fisher_mat]
    
    F_theta1 = {key: fisher_mat[1].get(key, 1.) * theta_1[key] for key in theta_1.keys()}
    F_theta0 = {key: fisher_mat[0].get(key, 1.)* theta_0[key] for key in theta_0.keys()}
    
    new_F = {
        key: ((1 - alpha) * fisher_mat[0][key] + alpha * fisher_mat[1][key])
        for key in fisher_mat[0].keys()
    }

    theta = {
        key: ((1 - alpha) * F_theta0[key] + alpha * F_theta1[key]) / new_F.get(key, 1.)
        for key in theta_0.keys() 
    }
    return theta, new_F

def finetune(args):
    train_dataset = args.dataset
    ckpdir = args.save

    for split_idx in range(args.n_splits):
        print(f"\n##### SPLIT {split_idx} #####")
        ft_path = os.path.join(ckpdir, f'finetuned_{split_idx}.pt')
        if os.path.exists(ft_path):
            continue
            
        if args.load is not None and args.load.endswith('pt'):
            image_encoder = LinearizedImageEncoder.load(args.load)
        elif args.sequential_finetuning and split_idx != 0:
            prev_ckpt = os.path.join(ckpdir, f'finetuned_{split_idx-1}.pt')
            image_encoder = LinearizedImageEncoder.load(prev_ckpt)
            prev_fisher = torch.load(os.path.join(ckpdir, f'fisher_{split_idx-1}.pt'))
        else:
            image_encoder = LinearizedImageEncoder(args)

        preprocess_fn = image_encoder.train_preprocess
        dataset = get_dataset(train_dataset, preprocess_fn, location=args.data_location, batch_size=args.batch_size)
        dataset, classification_head = get_dataset_and_classifier_for_split(dataset, split_idx, image_encoder, args)
        
        prev_image_encoder = copy.deepcopy(image_encoder)
        model = ImageClassifier(image_encoder, classification_head)
        model.freeze_head()
        model = model.to("cuda:0")
        
        loss_fn = LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        
        num_batches = len(dataset.train_loader)
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)

        for epoch in range(args.epochs):
            model.train()
            for i, batch in enumerate(data_loader):
                optimizer.zero_grad()
                batch = maybe_dictionarize(batch)
                inputs, labels = batch['images'].to('cuda:0'), batch['labels'].to('cuda:0')
                loss = loss_fn(model(inputs), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler(i + epoch * num_batches)

        # Collect FIM
        fisher = {name: state['exp_avg_sq'].clone().detach() 
                  for name, param in model.image_encoder.named_parameters() 
                  if param.requires_grad and 'exp_avg_sq' in optimizer.state.get(param, {})}

        state_dict = model.image_encoder.state_dict()
        if split_idx > 0:
            # INTEGRATED ADAPTIVE MERGING
            if args.alpha_mode == "adaptive":
                alpha, F_prev, F_curr = compute_adaptive_alpha(prev_fisher, fisher, args.alpha_min, args.alpha_max)
                log_adaptive_alpha("results/adaptive_alpha_cil.csv", split_idx, F_prev, F_curr, alpha)
                print(f"Adaptive Alpha: {alpha:.4f} (F_prev={F_prev:.2f}, F_curr={F_curr:.2f})")
            else:
                alpha = args.alpha_merge
            
            state_dict, fisher = interpolate_weights(prev_image_encoder.state_dict(), state_dict, alpha, [prev_fisher, fisher])
        
        model.image_encoder.load_state_dict(state_dict)
        
        # Representation Finetuning
        if split_idx > 0:
            model.train()
            prev_model = ImageClassifier(prev_image_encoder, classification_head).to("cuda:0").eval()
            opt = torch.optim.AdamW([model.image_encoder.get_trainable_params()], lr=args.representation_lr)
            for i, batch in enumerate(data_loader):
                model.zero_grad()
                inputs = maybe_dictionarize(batch)['images'].to('cuda:0')
                loss = torch.nn.L1Loss()(model(inputs, return_features=True)[1], prev_model(inputs, return_features=True)[1])
                loss.backward()
                opt.step()
        
        # Save
        result = eval_single_dataset(model.image_encoder, args.dataset, args)
        if args.save:
            torch.save(fisher, os.path.join(ckpdir, f'fisher_{split_idx}.pt'))
            model.image_encoder.save(ft_path)

if __name__ == '__main__':
    args = parse_arguments()
    args.save = "outputs/cil/linear/" + str(args).replace(", ", "/").replace("'", "").replace("(", "").replace(")", "").replace("Namespace", "")
    os.makedirs(args.save, exist_ok=True)
    wandb.init(project="CIL-linear", config=vars(args))
    finetune(args)