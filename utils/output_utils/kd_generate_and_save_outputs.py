import pandas as pd
import torch
from tqdm import tqdm



def generate_evaluation_embeddings(model, dataloader, device):
    model.eval()
    output_embeds_list = []
    target_embeds_list = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc='Extracting Embeddings (Evaluate) Loop')):
            cxrs, target_embeds = batch['cxr'].to(device), batch['embedding'].to(device)
            output_embeds = model(cxrs)
            output_embeds_list.append(output_embeds)
            target_embeds_list.append(target_embeds)

        output_embeds_array = torch.cat(output_embeds_list, dim=0)
        target_embeds_array = torch.cat(target_embeds_list, dim=0)

    return target_embeds_array.cpu().numpy(), output_embeds_array.cpu().numpy()


def save_embeddings_to_csv(embeds, target_embeds, file_path):
    cols_names_embeds = [f'embed_{i+1}' for i in range(embeds.shape[1])]
    cols_names_target_embeds = [f'target_embed_{i+1}' for i in range(target_embeds.shape[1])]
    
    df_embeddings = pd.DataFrame(data=embeds, columns=cols_names_embeds)
    df_targets = pd.DataFrame(data=target_embeds, columns=cols_names_target_embeds)
    df = pd.concat([df_embeddings, df_targets], axis=1)
    df.to_csv(file_path, index=False)


def run_evaluation_phase(model, dataloader, device, file_path, phase):
    print(f'<<>> {phase.upper()} PHASE <<>>')
    if 'embeddings' in phase:
        # model.remove_head()
        target_embeds, pre_embeds = generate_evaluation_embeddings(model=model, dataloader=dataloader, device=device)
        save_embeddings_to_csv(embeddings=pre_embeds, targets=target_embeds, file_path=file_path)
    else:
        target_embeds, output_embeds = generate_evaluation_embeddings(model=model, dataloader=dataloader, device=device)
        save_embeddings_to_csv(embeddings=output_embeds, targets=target_embeds, file_path=file_path)


