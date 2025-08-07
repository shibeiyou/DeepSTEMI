import torch

class Config:
    cine_shape = (25, 224, 224) 
    t2_shape = (224, 224)
    lge_shape = (224, 224)
    
    video_swin_embed_dim = 96
    swin_embed_dim = 96
    mlp_hidden_dim = 256
    mlp_output_dim = 768
    num_classes = 2
    
    batch_size = 2
    num_workers = 4
    learning_rate = 1e-4
    weight_decay = 1e-4
    epochs = 50
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')