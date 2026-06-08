import copy
import os
import time
import torch
import wandb
from torchinfo import summary
import yaml

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset, registry
from src.eval import eval_single_dataset, eval_given_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
# Import adaptive utility
from src.utils2.merge_utils import compute_adaptive_alpha, log_adaptive_alpha

PRINT_EVERY = 100

def interpolate_weights(theta_0, theta_1, alpha, fisher_mat=None):
    theta_0 = {k:v.cpu() for k,v in theta_0.items() if isinstance(v, torch.Tensor)}
    theta_1 = {k:v.cpu() for k,v in theta_1.items() if isinstance(v, torch.Tensor)}
    
    if fisher_mat is None:
        theta = {key: ((1 - alpha) * theta_0[key] + alpha * theta_1[key]) for key in theta_0.keys()}
        return theta, None
    
    fisher_mat = [{k:v.cpu() for k,v in f.items() if isinstance(v, torch.Tensor) } for f in fisher_mat]
    F_theta1 = {key: fisher_mat[1].get(key, 1.) * theta_1[key] for key in theta_1.keys()}
    F_theta0 = {key: fisher_mat[0].get(key, 1.) * theta_0[key] for key in theta_0.keys()}
    
    new_F = {key: ((1 - alpha) * fisher_mat[0][key] + alpha * fisher_mat[1][key]) for key in fisher_mat[0].keys()}

    theta = {
        key: ((1 - alpha) * F_theta0[key] + alpha * F_theta1[key]) / new_F.get(key, 1.)
        for key in theta_0.keys() 
    }
    return theta, new_F

def finetune(args, eval_0shot=False, only_eval_0shot=False):
    train_dataset_name = args.dataset
    dataset_class = registry[train_dataset_name].BASE_CLASS
    ckpdir = args.save
    subset_config_id = dataset_class.get_md5(args.subset_config)

    ft_path = os.path.join(ckpdir, f'checkpoint_{args.task_idx}.pt') if args.sequential_finetuning else os.path.join(ckpdir, f'checkpoint_{subset_config_id}.pt')
    if os.path.exists(ft_path):
        return

    if args.load is not None and args.load.endswith('pt'):
        image_encoder = LinearizedImageEncoder.load(args.load)
    elif args.sequential_finetuning and args.task_idx:
        prev_ckpt = os.path.join(ckpdir, f'checkpoint_{args.task_idx-1}.pt')
        image_encoder = LinearizedImageEncoder.load(prev_ckpt)
        prev_fisher = torch.load(os.path.join(ckpdir, f'fisher_{args.task_idx-1}.pt'))
    else:
        image_encoder = LinearizedImageEncoder(args)        

    dataset = get_dataset(train_dataset_name, image_encoder.train_preprocess, location=args.data_location, batch_size=args.batch_size, subset_config=args.subset_config)
    
    prev_image_encoder = copy.deepcopy(image_encoder)
    classification_head = get_classification_head(args, train_dataset_name, classnames=dataset.classnames)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    model = model.cuda()

    loss_fn = LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * len(dataset.train_loader))

    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(get_dataloader(dataset, is_train=True, args=args)):
            optimizer.zero_grad()
            batch = maybe_dictionarize(batch)
            loss = loss_fn(model(batch['images'].cuda()), batch['labels'].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler(i + epoch * len(dataset.train_loader))

    # Collect FIM
    fisher = {name: state['exp_avg_sq'].clone().detach() for name, param in model.image_encoder.named_parameters() 
              if param.requires_grad and 'exp_avg_sq' in optimizer.state.get(param, {})}
    
    state_dict = model.image_encoder.state_dict()
    if args.task_idx > 0:
        # INTEGRATED ADAPTIVE MERGE
        if args.alpha_mode == "adaptive":
            alpha, F_prev, F_curr = compute_adaptive_alpha(prev_fisher, fisher, args.alpha_min, args.alpha_max)
            log_adaptive_alpha("results/adaptive_alpha_dil.csv", args.task_idx, F_prev, F_curr, alpha)
        else:
            alpha = args.alpha_merge
            
        state_dict, fisher = interpolate_weights(prev_image_encoder.state_dict(), state_dict, alpha=alpha, fisher_mat=[prev_fisher, fisher])
    
    model.image_encoder.load_state_dict(state_dict)

    # Representation Finetuning
    if args.task_idx > 0:
        model.train()
        prev_model = ImageClassifier(prev_image_encoder, classification_head).cuda().eval()
        opt = torch.optim.AdamW([model.image_encoder.get_trainable_params()], lr=args.representation_lr)
        for i, batch in enumerate(get_dataloader(dataset, is_train=True, args=args)):
            model.zero_grad()
            loss = torch.nn.L1Loss()(model(maybe_dictionarize(batch)['images'].cuda(), return_features=True)[1], prev_model(maybe_dictionarize(batch)['images'].cuda(), return_features=True)[1])
            loss.backward()
            opt.step()
            wandb.log({f"task_{args.task_idx}_representation_loss": loss.item()})

    # Save and Eval
    if args.save:
        image_encoder.save(ft_path)
        torch.save(fisher, os.path.join(ckpdir, f'fisher_{args.task_idx}.pt'))
    
    if not args.skip_eval:
        _full_r = eval_single_dataset(model.image_encoder, train_dataset_name, args)['top1']
        wandb.log({'full_acc': _full_r * 100.0, "task_idx": args.task_idx})

if __name__ == '__main__':
    args = parse_arguments()
    args.model, args.batch_size = 'ViT-B-16', 128
    args.save = "outputs/dil/linear/linear-interpolate/" + str(args).replace(", ", "/").replace("'", "").replace("(", "").replace(")", "").replace("Namespace", "")
    os.makedirs(args.save, exist_ok=True)
    wandb.init(project="DIL-linear", config=vars(args))
    for task_idx, domain_idx in enumerate(registry[args.dataset].default_domain_order):
        args.subset_config = {'domains': [registry[args.dataset].BASE_CLASS.DOMAINS[domain_idx]], 'classes': registry[args.dataset].BASE_CLASS.CLASSES, 'domain_idx': domain_idx}
        args.task_idx = task_idx if args.sequential_finetuning else None
        finetune(args)