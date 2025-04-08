import torch
from tqdm.auto import tqdm

def accFn(GTList, PredList):
    tp = 0
    
    for i in range(len(GTList)):
        if GTList[i] == PredList[i]:
            tp += 1
    
    return tp / len(GTList)


def train_step(model, dataloader, device, optimizer, loss_fn, acc_fn):
    
    train_loss, train_acc = 0, 0
    model.train()
    
    for batch, (X,y) in tqdm(enumerate(dataloader)):
        
        X, y = X.to(device), y.to(device)
        
        y_logits = model(X)
        y_pred = torch.argmax(y_logits, dim=-1)
        
        loss = loss_fn(y_logits, y)
        acc = acc_fn(y_pred, y)
        
        train_loss += loss
        train_acc += acc
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return {'model_name': model.__class__.__name__,
            'loss': train_loss,
            'acc': train_acc}



def test_step(model, dataloader, device, loss_fn, acc_fn):
    
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        
        for batch, (X,y) in tqdm(enumerate(dataloader)):
            
            X, y = X.to(device), y.to(device)
            
            y_logits = model(X)
            y_pred = torch.argmax(y_logits, dim=-1)
            
            loss = loss_fn(y_logits, y)
            acc = acc_fn(y_pred, y)
            
            test_loss += loss
            test_acc += acc
        
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        
        return {'model_name': model.__class__.__name__,
                'loss': test_loss,
                'acc': test_acc}
        
        
        
        

