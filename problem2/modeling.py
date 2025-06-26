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
        x = self.df.iloc[idx].drop(labels=[self.y_col]).to_numpy()
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
    import numpy as np
    class FourierLayer(torch.nn.Module):
        def __init__(self, pred_len, k_seasonal):
            super().__init__()
            self.pred_len = pred_len
            self.k = k_seasonal
            frequencies = torch.arange(1, k_seasonal + 1, dtype=torch.float32)
            self.register_buffer('frequencies', frequencies)

        def forward(self, seq_len):
            t = torch.arange(seq_len, dtype=torch.float32).to(self.frequencies.device)
            args = 2 * np.pi * self.frequencies[:, None] * t[None, :] / seq_len
            s_t = torch.sin(args)
            c_t = torch.cos(args)
            return torch.cat([s_t, c_t], dim=0).T

    class ETSLayer(torch.nn.Module):
        def __init__(self, d_model, seq_len, k_seasonal, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.seq_len = seq_len
            self.k = k_seasonal
            
            self.alpha = torch.nn.Parameter(torch.rand(1))
            self.smooth = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model),
                torch.nn.ReLU(),
                torch.nn.Linear(d_model, d_model)
            )

            self.beta = torch.nn.Parameter(torch.rand(1))
            self.growth = torch.nn.Linear(d_model, d_model)
            
            self.gamma = torch.nn.Parameter(torch.rand(1))
            self.fourier = FourierLayer(pred_len=seq_len, k_seasonal=k_seasonal)
            self.seasonal = torch.nn.Linear(2 * k_seasonal, d_model)

            self.dropout = torch.nn.Dropout(dropout)
        
        def forward(self, x):
            smooth_x = self.smooth(x)
            
            growth_x = self.growth(x)
            
            fourier_terms = self.fourier(self.seq_len)
            seasonal_x = self.seasonal(fourier_terms)
            
            x = self.alpha * smooth_x + \
                self.beta * growth_x + \
                self.gamma * seasonal_x[None, :, :] + \
                (1 - self.alpha - self.beta - self.gamma) * x
            
            return self.dropout(x)

    class ETSformer(torch.nn.Module):
        def __init__(self, input_features, seq_len, d_model, n_layers, k_seasonal, dropout):
            super().__init__()
            self.embedding = torch.nn.Linear(input_features, d_model)
            self.layers = torch.nn.ModuleList(
                [ETSLayer(d_model, seq_len, k_seasonal, dropout) for _ in range(n_layers)]
            )
            self.flatten = torch.nn.Flatten(start_dim=1)
            self.decoder = torch.nn.Linear(seq_len * d_model, 1)

        def forward(self, x):
            x = self.embedding(x)
            
            for layer in self.layers:
                x = layer(x)
                
            output = self.flatten(x)
            output = self.decoder(output)
            return output

    class ETSformerWrapper(torch.nn.Module):
        def __init__(self, actual_model: torch.nn.Module, seq_len: int):
            super().__init__()
            self.actual_model = actual_model
            self.seq_len = seq_len

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(1)
            x = x.repeat(1, self.seq_len, 1)
            return self.actual_model(x)

    INPUT_FEATURES = 15
    SEQ_LEN = 100
    
    K_SEASONAL = 10 
    
    D_MODEL = 64
    N_LAYERS = 2
    DROPOUT = 0.1

    etsformer_model = ETSformer(
        input_features=INPUT_FEATURES,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        k_seasonal=K_SEASONAL,
        dropout=DROPOUT
    )
    
    model = ETSformerWrapper(actual_model=etsformer_model, seq_len=SEQ_LEN)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
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
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
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

    epochs = 20
    patience = 20

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print("--- Training Started ---")
    for epoch_i in range(epochs):
        model.train()
        running_train_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_i, (x, y) in enumerate(train_loader):
            if (batch_i + 1) % 50 == 0:
                print(f"  Epoch {epoch_i+1}/{epochs} | Processing batch {batch_i+1}/{num_batches}")

            x, y = x.to(device), y.to(device)

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

        if patience_counter >= patience:
            print(f"\n--- Early stopping triggered after {patience} epochs with no improvement. ---")
            break

    if best_model_state:
        print(f"\n--- Loading best model from epoch with Val Loss: {best_val_loss:.4f} ---")
        model.load_state_dict(best_model_state)

    print("--- Training Finished ---")
    return model