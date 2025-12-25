import torch
import torch.nn as nn
import torch.nn.functional as F

class BehavioralCortex(nn.Module):
    def __init__(self, input_size=128, num_heads=4, num_classes=3):
        super(BehavioralCortex, self).__init__()
        
        # 1. Rhythm Detection: จับจังหวะชีพจรของระบบ
        self.rhythm_attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
        
        # 2. Intent Classifier: ระบุเจตนา (Aggression, Resistance, Genesis)
        self.intent_classifier = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes) 
        )
        
        # 3. Karma Predictor: ทำนายความรุ่ง/ร่วง ของอนาคต (0.0 - 1.0)
        self.karma_predictor = nn.Sequential(
            nn.Linear(input_size + num_classes, 32),
            nn.Sigmoid(), 
            nn.Linear(32, 1) 
        )

    def forward(self, temporal_features):
        # Rhythm Analysis
        rhythm_out, _ = self.rhythm_attention(temporal_features, temporal_features, temporal_features)
        context_vector = torch.mean(rhythm_out, dim=1) 
        
        # Intent Identification
        intent_logits = self.intent_classifier(context_vector)
        intent_probs = F.softmax(intent_logits, dim=1)
        
        # Karma Prediction
        karma_input = torch.cat((context_vector, intent_probs), dim=1)
        future_outcome = self.karma_predictor(karma_input)
        
        return intent_probs, future_outcome
