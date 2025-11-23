import torch
import numpy as np
from models.neurosparse import NeuroSparseTransformer

class MultimodalStreamProcessor:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = NeuroSparseTransformer().to(device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.eval()
        
    def process_stream(self, visual_frames, audio_data, imu_data):
        with torch.no_grad():
            visual_tensor = torch.FloatTensor(visual_frames).unsqueeze(0).to(self.device)
            audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0).to(self.device)
            imu_tensor = torch.FloatTensor(imu_data).unsqueeze(0).to(self.device)
            
            logits = self.model(visual_tensor, audio_tensor, imu_tensor)
            predictions = torch.softmax(logits, dim=1)
            
            return predictions.cpu().numpy()
    
    def process_single_frame(self, visual_frame, audio_frame, imu_frame):
        return self.process_stream(
            visual_frame.unsqueeze(0), 
            audio_frame.unsqueeze(0), 
            imu_frame.unsqueeze(0)
        )
