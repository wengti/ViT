
import torch
import yaml
from pathlib import Path
import json
from custom_data import custom_dataset
from matplotlib import pyplot as plt
from model import vision_transformer
from google.colab import drive
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
from engine import train_step, test_step, accFn
from tqdm.auto import tqdm
import pickle
import numpy as np

# 0. setup


# Reading configs
configPath = Path('./default.yaml')
with open(configPath, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


# Reading database
jsonPath = Path('./imdb.json')
with open(jsonPath, 'r') as f:
    jsonFile = json.load(f)



# Setting seeds and device
torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'




# Flag
saveInDrive = True
showTest = True
loadModel = True


# Variables
loadModelPath = Path("./best.pt")
loadTrainInfoPath = Path("./train_info.pt")

EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_WORKERS = 2
PATIENCE = 3
FACTOR= 0.5

folderName = f"VIT_{config['task_name']}_2"



# Setting File Directories
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












# 1. Test the loaded files
print(f"[INFO] The loaded config is as following: ")
for key, val in config.items():
    print(f"{key}: {val}")


# 2. Load the dataset
trainData = custom_dataset(config = config,
                           json_file = jsonFile,
                           split = 'train')

testData = custom_dataset(config = config,
                          json_file = jsonFile,
                          split = 'test')

# 3. Test the loaded dataset

if showTest:
    dataset = trainData

    randNum = torch.randint(0, len(dataset)-1, (9,))
    for idx, num in enumerate(randNum):
        trainImg, trainLabel = dataset[num]
        trainImgPlt = (trainImg + 1) / 2
        trainImgPlt = trainImgPlt.permute(1,2,0)

        plt.subplot(3,3, idx+1)
        plt.imshow(trainImgPlt)
        plt.axis(False)
        plt.title(f"Label: {trainLabel}")

    plt.tight_layout()
    plt.show()

    print(f"[INFO] The number of images in the dataset  : {len(dataset)}")
    print(f"[INFO] The size of an image in the dataset  : {trainImg.shape}")
    print(f"[INFO] The unique value within the dataset  : {trainImg.min()} to {trainImg.max()}")
    print(f"[INFO] The available classes in the dataset : {dataset.classes}")



# 4. Load the dataloader

trainDataLoader = DataLoader(dataset = trainData,
                             batch_size = BATCH_SIZE,
                             shuffle = True,
                             num_workers = NUM_WORKERS)

testDataLoader = DataLoader(dataset = testData,
                            batch_size = BATCH_SIZE,
                            shuffle = False,
                            num_workers = NUM_WORKERS)


# 5. Test the dataloader

if showTest:
    trainImgBatch, trainLabelBatch = next(iter(trainDataLoader))

    print(f"[INFO] The number of batches            : {len(trainDataLoader)}")
    print(f"[INFO] The number of images in a batch  : {trainImgBatch.shape[0]}")
    print(f"[INFO] The size of an image             : {trainImgBatch[0].shape}")




# 6. Create a model
model0 = vision_transformer(config = config).to(device)


# 7. Verify the model
# =============================================================================
# from torchinfo import summary
#
# summary(model = model0,
#         input_size = (1,3,224,224),
#         col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
#         row_settings = ['var_names'])
# =============================================================================



# 8. Create optimizer, scheduler and loss functions
optimizer = torch.optim.Adam(params = model0.parameters(),
                             lr = LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                       patience = PATIENCE,
                                                       factor = FACTOR)

lossFn = nn.CrossEntropyLoss()





# 9. Create training loop

trainLossList = []
testLossList = []
trainAccList = []
testAccList = []
bestLoss = np.inf

lossAccFile = resultFolder / 'loss_acc.pkl'
bestModelFile = resultFolder / 'best.pt'
trainInfoFile = resultFolder / 'train_info.pt'


# Load models
if loadModel:
    model0.load_state_dict(torch.load(f = loadModelPath,
                                              weights_only = True))
    model0 = model0.to(device)

    trainCheckPoint = torch.load(loadTrainInfoPath)
    optimizer.load_state_dict(trainCheckPoint['optimizer'])
    scheduler.load_state_dict(trainCheckPoint['scheduler'])

    # Show previous training progress
    if showTest:
        print("[INFO] At previous best model ")
        print(f"[INFO] Epoch          : {trainCheckPoint['epoch']}")
        print(f"[INFO] Train Loss     : {trainCheckPoint['train_loss']:.4f}")
        print(f"[INFO] Train Accuracy : {trainCheckPoint['train_acc']*100:.2f}%")
        print(f"[INFO] Test Loss      : {trainCheckPoint['test_loss']:.4f}")
        print(f"[INFO] Test Accuracy  : {trainCheckPoint['test_acc']*100:.2f}%")
        print(f"[INFO] Learning rate for this epoch: {trainCheckPoint['lr']}")




for epoch in tqdm(range(EPOCHS)):

    # Main training step
    trainResult = train_step(model = model0,
                             device = device,
                             dataloader = trainDataLoader,
                             optimizer = optimizer,
                             acc_fn = accFn,
                             loss_fn = lossFn)

    testResult = test_step(model = model0,
                           device = device,
                           dataloader = testDataLoader,
                           acc_fn = accFn,
                           loss_fn = lossFn)


    # Annoucing train results for this epoch
    trainLoss = trainResult['loss']
    trainAcc = trainResult['acc']
    testLoss = testResult['loss']
    testAcc = testResult['acc']

    print(f'[DEBUG] Scheduler: {scheduler}')

    latestLR = scheduler.get_last_lr()

    print(f"[INFO] Current Epoch: {epoch}")
    print(f"[INFO] Train Loss     : {trainLoss:.4f}")
    print(f"[INFO] Train Accuracy : {trainAcc*100:.2f}%")
    print(f"[INFO] Test Loss      : {testLoss:.4f}")
    print(f"[INFO] Test Accuracy  : {testAcc*100:.2f}%")
    print(f"[INFO] Learning rate for this epoch: {latestLR}")


    # Check if there's improvement and if a best model need to be saved
    if trainLoss < bestLoss:
        print(f"[UPDATE] The best loss has been improved from {bestLoss} to {trainLoss}.")
        print(f"[UPDATE] Proceed to save this as the best model at {bestModelFile}.")
        print(f"[UPDATE] The corresponding training info are saved at {trainInfoFile}")
        bestLoss = trainLoss
        torch.save(obj = model0.state_dict(),
                   f = bestModelFile) # Save the best model

        trainInfo = {'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     'epoch': epoch,
                     'train_loss': trainLoss,
                     'test_loss': testLoss,
                     'train_acc': trainAcc,
                     'test_acc': testAcc,
                     'lr': latestLR}
        torch.save(obj = trainInfo,
                   f = trainInfoFile) # Save the training info


    # Update the scheduler
    scheduler.step(trainLoss)


    # Save loss and accuracy into a pickle file
    trainLossList.append(trainLoss)
    trainAccList.append(trainAcc)
    testLossList.append(testLoss)
    testAccList.append(testAcc)

    lossAccDict = {'train_loss': trainLossList,
                   'train_acc': trainAccList,
                   'test_loss': testLossList,
                   'test_acc': testAccList}

    with open(lossAccFile, 'wb') as f:
        pickle.dump(lossAccDict, f)
