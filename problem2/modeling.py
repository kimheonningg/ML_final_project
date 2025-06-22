from typing import Optional
import torch
from torch.utils.data import Dataset
import pandas as pd


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class PandasDataset(Dataset):
    def __init__(self, df: pd.DataFrame, y_col: str):
        assert y_col in df.columns, f"y_col {y_col} not in df columns"
        self.df = df
        self.y_col = y_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx].drop(columns=[self.y_col]).to_numpy()
        y = self.df.iloc[idx][self.y_col]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


############################################################################################
"""

Model:

"""
############################################################################################


def sample_load_model() -> torch.nn.Module:
    model = torch.nn.Sequential(
        torch.nn.Linear(12, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )
    assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
    return model


def sampe_type_2_load_model() -> torch.nn.Module:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(12, 128)
            self.fc2 = torch.nn.Linear(128, 128)
            self.fc3 = torch.nn.Linear(128, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    model = Model()

    assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
    return model


def load_model() -> torch.nn.Module:
    model: torch.nn.Module = None
    ############################################################################################
    #
    #                   Feel free to modify the code below.
    #
    ############################################################################################
    import math
    class PositionalEncoding(torch.nn.Module):
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)

    class TransformerModel(torch.nn.Module):
        def __init__(self, input_features: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.2):
            super().__init__()
            self.model_type = 'Transformer'
            self.d_model = d_model
            self.encoder = torch.nn.Linear(input_features, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout)
            encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
            self.decoder = torch.nn.Linear(d_model, 1)
            self.init_weights()

        def init_weights(self) -> None:
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)

        def forward(self, src: torch.Tensor) -> torch.Tensor:
            if src.dim() == 2:
                src = src.unsqueeze(1)
            src = self.encoder(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src)
            output = output.squeeze(1)
            output = self.decoder(output)
            return output

    INPUT_FEATURES = 16
    D_MODEL = 128
    N_HEAD = 8
    D_HID = 256
    N_LAYERS = 3
    DROPOUT = 0.2
    model = TransformerModel(input_features=INPUT_FEATURES, d_model=D_MODEL, nhead=N_HEAD, d_hid=D_HID, nlayers=N_LAYERS, dropout=DROPOUT)
    
    # TODO: implement your own model after removing this line
    ############################################################################################
    assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
    return model


############################################################################################
"""

Optimizer and Scheduler:

"""
############################################################################################


def sample_load_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    assert isinstance(optimizer, torch.optim.Optimizer), (
        "optimizer must be a torch.optim.Optimizer"
    )
    return optimizer


def load_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer = None
    ############################################################################################
    #
    #                   Feel free to modify the code below.
    #
    ############################################################################################
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    return optimizer


def sample_load_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.95**epoch
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler), (
        "scheduler must be a torch.optim.lr_scheduler.LambdaLR"
    )
    return scheduler


def load_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler = None
    ############################################################################################
    #
    #                   Feel free to modify the code below.
    #
    ############################################################################################
    epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return scheduler


############################################################################################
"""

train

"""
############################################################################################


def sample_train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    device = device or get_device()
    model.to(device=device)
    n_iter = 0

    for epeoch_i in range(3):
        model.train()
        for batch_i, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(device=device)
            y = y.to(device=device)
            optimizer.zero_grad()
            pred_y = model(x)
            loss = torch.nn.functional.mse_loss(pred_y.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()
            n_iter += 1
            if n_iter % 1000 == 0:
                print(f"Epoch {epeoch_i}, Iter {n_iter}, Loss {loss.item():.4f}")

        model.eval()
        val_loss = []
        val_n = 0
        print("start validation")
        for batch in val_loader:
            x, y = batch
            x = x.to(device=device)
            y = y.to(device=device)
            pred_y = model(x)
            loss = torch.nn.functional.mse_loss(pred_y.squeeze(), y.squeeze())
            val_loss.append(loss.item())
            val_n += x.shape[0]
        print(f"Epoch {epeoch_i}, Loss {sum(val_loss) / val_n}")

    return model


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    ############################################################################################
    #
    #                   Feel free to modify the code below.
    #
    ############################################################################################
    import copy

    device = device or get_device()
    model.to(device=device)

    epochs = 30
    patience = 7

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print("--- Training Started ---")
    for epoch_i in range(epochs):
        model.train()
        running_train_loss = 0.0
        for batch_i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            if x.dim() == 2:
                x = x.unsqueeze(1) 

            pred_y = model(x)
            loss = torch.nn.functional.mse_loss(pred_y.squeeze(), y.squeeze())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                if x.dim() == 2:
                    x = x.unsqueeze(1)

                pred_y = model(x)
                loss = torch.nn.functional.mse_loss(pred_y.squeeze(), y.squeeze())
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch {epoch_i+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> Found new best model with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  -> Val Loss did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\n--- Early stopping triggered ---")
            break

    if best_model_state:
        print(f"\n--- Loading best model from epoch with Val Loss: {best_val_loss:.4f} ---")
        model.load_state_dict(best_model_state)

    print("--- Training Finished ---")
    return model