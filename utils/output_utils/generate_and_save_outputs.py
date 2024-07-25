import pandas as pd
import torch
from tqdm import tqdm



def generate_evaluation_outputs(model, dataloader, device, num_classes, input_type='cxr'):
    model.eval()
    logits_list = []
    probs_list = []
    targets_list = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc='Evaluate Loop')):
            if input_type not in batch:
                raise KeyError(f"{input_type} is not a valid key in batch. Valid keys are: {list(batch.keys())}")
            
            inputs, labels = batch[input_type].to(device), batch['label'].to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            logits_list.append(logits)
            probs_list.append(probs)
            targets_list.append(labels)

        logits_array = torch.cat(logits_list, dim=0)
        probs_array = torch.cat(probs_list, dim=0)
        targets_array = torch.cat(targets_list, dim=0)

        counts = []
        for i in range(num_classes):
            t = (targets_array[:, i] == 1)
            c = torch.sum(t)
            counts.append(c)
        print("Class counts:", counts)

    return targets_array.cpu().numpy(), logits_array.cpu().numpy(), probs_array.cpu().numpy()


def generate_evaluation_embeddings(model, dataloader, device, input_type='cxr'):
    model.eval()
    embeddings_list = []
    targets_list = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc='Extracting Embeddings Loop')):
            if input_type not in batch:
                raise KeyError(f"{input_type} is not a valid key in batch. Valid keys are: {list(batch.keys())}")
            
            inputs, labels = batch[input_type].to(device), batch['label'].to(device)
            embeddings = model(inputs)
            embeddings_list.append(embeddings)
            targets_list.append(labels)

        embeddings_array = torch.cat(embeddings_list, dim=0)
        targets_array = torch.cat(targets_list, dim=0)

    return targets_array.cpu().numpy(), embeddings_array.cpu().numpy()


def save_outputs_to_csv(probs, logits, targets, num_classes, file_path):
    cols_names_probs = [f'prob_class_{i}' for i in range(num_classes)]
    cols_names_logits = [f'logit_class_{i}' for i in range(num_classes)]
    cols_names_targets = [f'target_class_{i}' for i in range(num_classes)]
    
    df_probs = pd.DataFrame(data=probs, columns=cols_names_probs)
    df_logits = pd.DataFrame(data=logits, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df = pd.concat([df_logits, df_probs, df_targets], axis=1)
    df.to_csv(file_path, index=False)


def save_embeddings_to_csv(embeddings, targets, num_classes, file_path):
    cols_names_embeddings = [f'embed_{i}' for i in range(embeddings.shape[1])]
    cols_names_targets = [f'target_class_{i}' for i in range(num_classes)]
    
    df_embeddings = pd.DataFrame(data=embeddings, columns=cols_names_embeddings)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df = pd.concat([df_embeddings, df_targets], axis=1)
    df.to_csv(file_path, index=False)


def run_evaluation_phase(model, dataloader, device, num_classes, file_path, phase, input_type):
    print(f'<<>> {phase.upper()} PHASE <<>>')
    if 'embeddings' in phase:
        model.remove_head()
        targets, embeddings = generate_evaluation_embeddings(model=model, dataloader=dataloader, device=device, 
                                                             input_type=input_type)
        save_embeddings_to_csv(embeddings=embeddings, targets=targets, num_classes=num_classes, file_path=file_path)
    else:
        targets, logits, probs = generate_evaluation_outputs(model=model, dataloader=dataloader, device=device, 
                                                             num_classes=num_classes, input_type=input_type)
        save_outputs_to_csv(probs=probs, logits=logits, targets=targets, num_classes=num_classes, file_path=file_path)