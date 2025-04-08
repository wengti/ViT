
import torch
from model import vision_transformer
import yaml
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import cv2
import numpy as np





def plot_cm(model, dataloader, save_folder):
    
    save_folder = Path(save_folder)
    save_file = save_folder / 'confusion_matrix.png'
    
    GTs = []
    Preds = []
    model.eval()
    with torch.inference_mode():
        
        for batch, (X,y) in tqdm(enumerate(dataloader)):
            
            GTs.append(y)
            X, y = X.to(device), y.to(device)
            
            y_logits = model(X)
            y_preds = torch.argmax(y_logits, dim=-1)

            Preds.append(y_preds.detach().cpu())
        
        GTs = torch.cat(GTs)
        Preds = torch.cat(Preds)
        
        cm = confusion_matrix(GTs, Preds)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot(cmap = "Blues")
        plt.title('Confusion Matrix')
        plt.savefig(save_file)
        plt.show()
        print(f"[INFO] The confusion matrix has been saved into {save_file}.")



def plot_pos_embedding(model, save_folder):
    save_folder = Path(save_folder)
    save_file = save_folder / 'positional_embedding.png'
    
    pos_embedding = model.patch_embedding.embedding_tokens[1:, :].detach().cpu() # N x D

    fig, axs = plt.subplots(7,7)
    axs = axs.flatten()

    count = 0
    for idx in range(len(pos_embedding)):
        
        row = idx // 14
        col = idx % 14
        
        if row%2 == 0 and col%2 == 0:
            
            similarity = torch.nn.functional.cosine_similarity(pos_embedding[idx], pos_embedding, dim=1) # N 
            similarity = similarity.reshape((14,14)) # Reshape to 14x14
            similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min()) # Scale from 0 to 1
            im = axs[count].imshow(similarity, vmin=0, vmax=1)
            axs[count].axis(False)
            count += 1
        
    fig.tight_layout()
    fig.colorbar(im, ax = axs.tolist())
    fig.savefig(save_file)
    fig.show()
    
    print(f"[INFO] The positional embedding plot has been saved into {save_file}")







def plot_attention_map(model, dataset, device, save_folder):
    
    numImages = 10
    randNum = torch.randint(0, len(dataset)-1, (numImages,))
    testImages = torch.cat([dataset[num][0][None,...] for num in randNum], dim=0) # B x C x H x W
    
    attentions = []
    def get_attention(model, inpt, outpt):
        attentions.append(outpt.detach().cpu())
    
    for name, module in model.named_modules():
        if 'att_map_dropout' in name:
            module.register_forward_hook(get_attention)
            
    model.eval()
    with torch.inference_mode():
        X = testImages.to(device)
        y_logits = model(X)
    
    
    # For each row, divide to elements with sum to 1
    # attention -> B x num_heads x (N+1) x (N+1)
    # attentions -> [attention at 1st layer, attention at 2nd layer.....]
    attentions = [(attention + torch.eye(attention.shape[-1])) / (attention + torch.eye(attention.shape[-1])).sum(dim=-1).unsqueeze(-1) 
                  for attention in attentions ] 
    
    query = attentions[0] # B x num_heads x (N+1) x (N+1)
    result = torch.mean(query, dim=1) # B x (N+1) x (N+1)
    for i in range(1,6):
        query = attentions[i]
        query = torch.mean(query, dim=1)
        result = torch.matmul(query, result)# B x (N+1) x (N+1)
    
    masks = result[:, 0, 1:] # B x N
    for idx in range(numImages):
        mask = masks[idx]
        mask = mask.numpy().reshape((14,14)) # 14 x 14, array
        mask = cv2.resize(mask, (224,224), interpolation = cv2.INTER_LINEAR)[..., None] # 224 x 224 x 1, array
        mask = (mask - mask.min()) / (mask.max() - mask.min()) # Scale mask value from 0 to 1, # 224 x 224 x 1, array float
        
        image = testImages[idx] # C x H x W, tensor, cpu, -1 to 1, float
        image = image.permute(1,2,0) # H x W x C, tensor, cpu, -1 to 1, float
        image = image.numpy() # H x W x C, numpy, cpu, -1 to 1, float
        image = np.uint8(((image + 1) / 2) * 255) # H x W x C, numpy, cpu, 0 to 255, integer
        
        masked_image = np.uint8(mask * image) # HxWxC, numpy, cpu, 0 to 255, integer
        
        save_folder = Path(save_folder)
        save_image_file = save_folder / f'Original_image_{idx}.png'
        save_mask_file = save_folder / f'Masked_image_{idx}.png'
        
        cv2.imwrite(save_image_file, image)
        cv2.imwrite(save_mask_file, masked_image)
        print(f"[INFO] Both original and masked images have been saved into {save_image_file} and {save_mask_file} respectively.")
        
    







device = 'cuda' if torch.cuda.is_available() else 'cpu'


configPath = Path('./default.yaml')
with open(configPath, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# Flag
saveInDrive = True
loadModel = True

# Variables
folderName = f"VIT_{config['task_name']}"
loadModelPath = "./best.pt"


if saveInDrive:
    drive.mount('/content/drive')
    gDriveFolder = Path(f'./drive/MyDrive/{folderName}')
    if not gDriveFolder.is_dir():
        gDriveFolder.mkdir(parents = True,
                           exist_ok = True)
    print(f"[INFO] Setting up save directories in Google Drive at {gDriveFolder}.")
    resultFolder = gDriveFolder
else:
    localFolder = Path(f'./{{folderName}}')
    if not localFolder.is_dir():
        localFolder.mkdir(parents = True,
                          exist_ok = True)
    print(f"[INFO] Setting up save directories in local device at {localFolder}.")
    resultFolder = localFolder




model1 = vision_transformer(config = config).to(device)
if loadModel == True:
  model1.load_state_dict(torch.load(f = loadModelPath,
                                    weights_only = True))
  model1 = model1.to(device)


# =============================================================================
# plot_cm(model = model1,
#         dataloader = testDataLoader,
#         save_folder = resultFolder)
# 
# 
# 
# plot_pos_embedding(model = model1,
#                     save_folder = resultFolder)
# 
# 
# 
# plot_attention_map(model = model1,
#                    dataset = testData,
#                    device = device,
#                    save_folder = resultFolder)
# =============================================================================
