import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F



def generate_evaluation_outputs(model, dataloader, device, num_classes, input_type='cxr'):
    model.eval()
    batch_indices = []
    logits_list = []
    probs_list = []
    targets_list = []
    losses_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluate Loop')):
            if input_type not in batch:
                raise KeyError(f"{input_type} is not a valid key in batch. Valid keys are: {list(batch.keys())}")
            
            inputs, labels = batch[input_type].to(device), batch['label'].to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            losses = F.binary_cross_entropy(probs, labels, reduction='none')  # No reduction yet
            losses_per_sample = losses.mean(dim=1)  # Mean across classes

            batch_indices.extend([batch_idx] * len(labels))
            logits_list.append(logits)
            probs_list.append(probs)
            targets_list.append(labels)
            losses_list.extend(losses_per_sample.tolist())

        logits_array = torch.cat(logits_list, dim=0)
        probs_array = torch.cat(probs_list, dim=0)
        targets_array = torch.cat(targets_list, dim=0)

        # Calculate class counts and frequencies
        counts = []
        frequencies = []
        total_samples = targets_array.shape[0]
        for i in range(num_classes):
            class_count = (targets_array[:, i] == 1).sum().item()
            counts.append(class_count)
            frequencies.append(class_count / total_samples if total_samples > 0 else 0)
        print("Class counts:", counts)
        print("Class frequencies:", frequencies)

    return batch_indices, targets_array.cpu().numpy(), logits_array.cpu().numpy(), probs_array.cpu().numpy(), losses_list


def generate_evaluation_embeddings(model, dataloader, device, input_type='cxr'):
    model.eval()
    batch_indices = []
    embeddings_list = []
    targets_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Extracting Embeddings Loop')):
            if input_type not in batch:
                raise KeyError(f"{input_type} is not a valid key in batch. Valid keys are: {list(batch.keys())}")
            
            inputs, labels = batch[input_type].to(device), batch['label'].to(device)
            embeddings = model(inputs)

            batch_indices.extend([batch_idx] * len(labels))
            embeddings_list.append(embeddings)
            targets_list.append(labels)

        embeddings_array = torch.cat(embeddings_list, dim=0)
        targets_array = torch.cat(targets_list, dim=0)

    return batch_indices, targets_array.cpu().numpy(), embeddings_array.cpu().numpy()


def save_outputs_to_csv(probs, logits, targets, losses, batch_indices, num_classes, file_path):
    cols_names_probs = [f'prob_class_{i+1}' for i in range(num_classes)]
    cols_names_logits = [f'logit_class_{i+1}' for i in range(num_classes)]
    cols_names_targets = [f'target_class_{i+1}' for i in range(num_classes)]
    cols_names_losses = ['individual_loss']
    cols_names_batches = ['batch_index']
    
    df_probs = pd.DataFrame(data=probs, columns=cols_names_probs)
    df_logits = pd.DataFrame(data=logits, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df_losses = pd.DataFrame(data=losses, columns=cols_names_losses)
    df_batches = pd.DataFrame(data=batch_indices, columns=cols_names_batches)

    df = pd.concat([df_batches, df_logits, df_probs, df_targets, df_losses], axis=1)
    df.to_csv(file_path, index=False)


def save_embeddings_to_csv(embeddings, targets, batch_indices, num_classes, file_path):
    cols_names_embeddings = [f'embed_{i+1}' for i in range(embeddings.shape[1])]
    cols_names_targets = [f'target_class_{i+1}' for i in range(num_classes)]
    cols_names_batches = ['batch_index']
    
    df_embeddings = pd.DataFrame(data=embeddings, columns=cols_names_embeddings)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df_batches = pd.DataFrame(data=batch_indices, columns=cols_names_batches)

    df = pd.concat([df_batches, df_embeddings, df_targets], axis=1)
    df.to_csv(file_path, index=False)


def run_evaluation_phase(model, dataloader, device, num_classes, file_path, phase, input_type):
    print(f'<<>> {phase.upper()} PHASE <<>>')
    if 'embeddings' in phase:
        # model.remove_head()
        batch_indices, targets, embeddings = generate_evaluation_embeddings(model=model, dataloader=dataloader, 
                                                                            device=device, input_type=input_type)
        save_embeddings_to_csv(embeddings=embeddings, targets=targets, batch_indices=batch_indices, 
                               num_classes=num_classes, file_path=file_path)
    else:
        batch_indices, targets, logits, probs, losses = generate_evaluation_outputs(model=model, dataloader=dataloader, 
                                                                                    device=device, num_classes=num_classes, 
                                                                                    input_type=input_type)
        save_outputs_to_csv(probs=probs, logits=logits, targets=targets, losses=losses, batch_indices=batch_indices, 
                            num_classes=num_classes, file_path=file_path)


