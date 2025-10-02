from nn.modelNet import Net
import torch
import os

def merge(static_model, new_static_model,alpha=0.2):
    try:
        new_w = torch.load(new_static_model, map_location="cpu")
        if not is_compatible(new_w):
            print("\n=====================\n",f"The model {new_static_model} is incompatible!\n=====================\n")
            new_w = None

    except Exception as e:
        print("\n=====================\n",f"ERROR in load new model {new_static_model}: \n")
        print(e,"\n\n=====================\n")
        new_w = None

    if not new_w:
        return False
    
    if os.path.exists(static_model):
        try:
            old_w = torch.load(static_model, map_location='cpu')
            merged = {}
            for k in old_w.keys():
                
                if k in new_w.keys() and old_w[k].shape == new_w[k].shape:
                    merged[k] = (1 - alpha)*old_w[k] + alpha*new_w[k]
                
                else:
                    merged[k] = old_w[k]
            
            torch.save(merged, static_model)
            print(f"Merged saved in {static_model}")
            return True
        
        except Exception as e:
            print("\n=========================\nMerged is failed!\nException: \n")
            print(e)
            print("\n=========================\n")
            return False
        
    else:
        torch.save(new_w, static_model)
        print(f"Old model doesnt exist! New model saved in {static_model}")
        return True

def is_compatible(static_dict):
    base_model = Net()
    model_dict = base_model.state_dict()

    missing = model_dict.keys() - static_dict.keys()
    unexpected = static_dict.keys() - model_dict.keys()

    if missing or unexpected:
        return False
    
    for k in model_dict.keys():
        if model_dict[k].shape != static_dict[k].shape:
            return False
        
    return True
        