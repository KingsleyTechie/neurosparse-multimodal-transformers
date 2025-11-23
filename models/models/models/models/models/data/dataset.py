import torch
from torch.utils.data import Dataset
import numpy as np
import math
import random
from dataclasses import dataclass

@dataclass
class MultimodalSample:
    visual_seq: torch.Tensor
    audio_seq: torch.Tensor
    imu_seq: torch.Tensor
    label: int
    timestamp: torch.Tensor

class MultimodalStreamGenerator:
    def __init__(self, seq_length=50, num_classes=5):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.actions = {
            0: "clapping",
            1: "walking", 
            2: "jumping",
            3: "waving",
            4: "standing"
        }
        
    def _generate_visual_pattern(self, action_id, position):
        frames = []
        for t in range(self.seq_length):
            frame = torch.zeros(3, 32, 32)
            
            if action_id == 0:
                hand_pos = int(10 + 5 * math.sin(2 * math.pi * t / 15 + position))
                frame[0, 10:15, hand_pos:hand_pos+5] = 1.0
                frame[0, 10:15, 25-hand_pos:30-hand_pos] = 1.0
                
            elif action_id == 1:
                leg_offset = int(3 * math.sin(2 * math.pi * t / 20 + position))
                frame[2, 20:25, 12+leg_offset:17+leg_offset] = 1.0
                frame[2, 20:25, 28-leg_offset:33-leg_offset] = 1.0
                
            elif action_id == 2:
                jump_height = int(8 * math.sin(math.pi * t / 25))
                frame[1, 15-jump_height:20-jump_height, 12:20] = 1.0
                
            elif action_id == 3:
                wave_pos = int(6 * math.sin(2 * math.pi * t / 10 + position))
                frame[0, 8:13, 25+wave_pos:30+wave_pos] = 1.0
                
            else:
                frame[1, 15:20, 12:20] = 0.7
            
            frames.append(frame)
        return torch.stack(frames)
    
    def _generate_audio_pattern(self, action_id, intensity):
        audio_features = []
        for t in range(self.seq_length):
            features = torch.zeros(40)
            
            if action_id == 0:
                if t % 15 < 2:
                    features[10:20] = intensity * (0.5 + 0.5 * torch.rand(10))
                    
            elif action_id == 1:
                if t % 10 < 3:
                    features[5:15] = intensity * 0.3 * (0.8 + 0.2 * torch.rand(10))
                    
            elif action_id == 2:
                if 20 <= t < 25:
                    features[15:25] = intensity * (0.7 + 0.3 * torch.rand(10))
                    
            elif action_id == 3:
                if t % 8 < 2:
                    features[2:8] = intensity * 0.2 * (0.5 + 0.5 * torch.rand(6))
            
            features += 0.05 * torch.randn(40)
            audio_features.append(features)
            
        return torch.stack(audio_features)
    
    def _generate_imu_pattern(self, action_id, amplitude):
        imu_data = []
        for t in range(self.seq_length):
            sample = torch.zeros(6)
            
            if action_id == 0:
                sample[0] = amplitude * math.sin(2 * math.pi * t / 15) + 0.1 * torch.randn(1)
                
            elif action_id == 1:
                sample[1] = amplitude * 0.5 * math.sin(2 * math.pi * t / 10)
                sample[5] = amplitude * 0.3 * math.sin(2 * math.pi * t / 10 + math.pi/2)
                
            elif action_id == 2:
                if 10 <= t < 15:
                    sample[2] = amplitude * 2.0
                elif 20 <= t < 25:
                    sample[2] = -amplitude * 1.5
                    
            elif action_id == 3:
                sample[3] = amplitude * math.sin(2 * math.pi * t / 8)
                sample[4] = amplitude * 0.5 * math.cos(2 * math.pi * t / 8)
            
            sample[2] += 1.0
            sample += 0.05 * torch.randn(6)
            imu_data.append(sample)
            
        return torch.stack(imu_data)
    
    def generate_sample(self, action_id=None):
        if action_id is None:
            action_id = random.randint(0, self.num_classes-1)
            
        position = random.uniform(0, 2 * math.pi)
        intensity = random.uniform(0.7, 1.3)
        amplitude = random.uniform(0.8, 1.2)
        
        visual_seq = self._generate_visual_pattern(action_id, position)
        audio_seq = self._generate_audio_pattern(action_id, intensity)  
        imu_seq = self._generate_imu_pattern(action_id, amplitude)
        timestamp = torch.linspace(0, 1, self.seq_length)
        
        return MultimodalSample(
            visual_seq=visual_seq,
            audio_seq=audio_seq,
            imu_seq=imu_seq,
            label=action_id,
            timestamp=timestamp
        )

class MultimodalDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=50, num_classes=5):
        self.generator = MultimodalStreamGenerator(seq_length, num_classes)
        self.num_samples = num_samples
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.generator.generate_sample()
        visual_flat = sample.visual_seq.view(sample.visual_seq.shape[0], -1)
        return visual_flat, sample.audio_seq, sample.imu_seq, sample.label
