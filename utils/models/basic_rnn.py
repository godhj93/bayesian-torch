import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_dnn(nn.Module):
    def __init__(self, vocab_size, emb_dim=10, hidden_dim=32, lstm_layers=1, output_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=False
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(x)
        h_final = hidden[-1]
        x = F.relu(self.fc1(h_final))
        logits = self.fc2(x)
        return logits


if __name__ == "__main__":
    # Example usage
    model = RNN_dnn(vocab_size=10000, emb_dim=300, hidden_dim=512, lstm_layers=1, output_dim=4)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # Dummy input
    input_ids = torch.randint(0, 10000, (32, 50))  # Batch size of 32, sequence length of 50
    logits = model(input_ids)
    print(logits.shape)  # Should be (32, 4) for the output dimension