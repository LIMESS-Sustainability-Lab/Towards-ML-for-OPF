import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import copy
from tqdm import tqdm

# ------------------------------
# Set seeds for reproducibility of experiments, however similar results are obtained for different seeds
# ------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
#             Utility Functions for Multi-Scenario
# =========================================================

def load_network(case_name):
    """Load a standard network from pandapower."""
    networks = {
        "case30": pn.case30(),
        "case_ieee30": pn.case_ieee30(),
        "case39": pn.case39(),
        "case118": pn.case118(),
        "GBreducednetwork": pn.GBreducednetwork(),
    }
    net = networks.get(case_name, None)
    if net is None:
        raise ValueError(f"Unknown case name: {case_name}")
    return net

def generate_opf_dataset(case_name, n_scenarios=300,
                         p_scale_range=(0.8, 1.2),
                         q_scale_range=(0.8, 1.2)):
    """
    1) Load the pandapower network
    2) For each scenario:
        - Randomly scale the load p_mw and q_mvar within provided ranges
        - runopp
        - Collect the entire system load as X (flattened)
        - Collect the entire generator results as Y (flattened)
    Returns (X, Y) arrays.
    You may get fewer than n_scenarios if OPF fails for certain extremes.
    """
    base_net = load_network(case_name)

    load_buses = base_net.load.index.values
    gen_buses = base_net.gen.index.values

    X_list = []
    Y_list = []

    # Add progress bar for scenario generation
    pbar = tqdm(total=n_scenarios, desc=f"Generating scenarios for {case_name}")
    successful_scenarios = 0

    for _ in range(n_scenarios):
        # Copy the base network so each scenario starts from original conditions
        net = copy.deepcopy(base_net)

        # Scale p_mw and q_mvar within random factors
        for lb in load_buses:
            rand_p_factor = np.random.uniform(*p_scale_range)
            rand_q_factor = np.random.uniform(*q_scale_range)
            net.load.at[lb, 'p_mw'] *= rand_p_factor
            net.load.at[lb, 'q_mvar'] *= rand_q_factor

        # Run OPF
        try:
            pp.runopp(net, verbose=False)
            successful_scenarios += 1
            
            # Collect data
            scenario_load_p = net.load['p_mw'].values
            scenario_load_q = net.load['q_mvar'].values
            X_scenario = np.hstack([scenario_load_p, scenario_load_q])

            scenario_gen_p = net.res_gen['p_mw'].values
            scenario_gen_vm = net.res_gen['vm_pu'].values
            Y_scenario = np.hstack([scenario_gen_p, scenario_gen_vm])

            X_list.append(X_scenario)
            Y_list.append(Y_scenario)
        except pp.optimal_powerflow.OPFNotConverged:
            pass
            
        pbar.update(1)
        pbar.set_postfix({'successful': successful_scenarios})

    pbar.close()

    X = np.array(X_list)
    Y = np.array(Y_list)

    return X, Y

# =========================================================
#           Data Scaling and Splitting
# =========================================================

def scale_data(inputs, outputs):
    """
    Apply standard scaling to inputs and outputs.
    Returns scaled arrays and the fitted scalers for potential inverse transform.
    """
    in_scaler = StandardScaler().fit(inputs)
    out_scaler = StandardScaler().fit(outputs)
    inputs_scaled = in_scaler.transform(inputs)
    outputs_scaled = out_scaler.transform(outputs)
    return inputs_scaled, outputs_scaled, in_scaler, out_scaler

def train_val_test_split(X, Y, test_size=0.15, val_size=0.15):
    """
    Split data into train, validation, and test sets.
    - First, do a train+val / test split
    - Then split train+val into train and val.

    Example: test_size=0.15, val_size=0.15 => total test=15%, val=15%, train=70%
    """
    X_tv, X_test, Y_tv, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=SEED
    )
    # Now split X_tv into train/val
    val_fraction_of_tv = val_size / (1.0 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tv, Y_tv, test_size=val_fraction_of_tv, random_state=SEED
    )
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# =========================================================
#               Dataset and Model Classes
# =========================================================

class OPFDataset(Dataset):
    """A simple torch Dataset for input-output pairs."""
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, x_size, y_size, hidden_sizes, dropout_rate=0.0):
        """
        hidden_sizes: list of hidden layer sizes, e.g. [256, 256] or [256, 256, 128]
        """
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = x_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_size = h
        # Output layer
        layers.append(nn.Linear(prev_size, y_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =========================================================
#             Training and Evaluation
# =========================================================

def train_model(model, train_loader, val_loader, epochs, learning_rate, patience=5):
    """
    Train the neural network with optional Early Stopping if val_loader is not None.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    use_early_stopping = (val_loader is not None)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Add progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training", leave=False)

    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        if use_early_stopping and val_loader is not None:
            val_loss = evaluate_model(model, val_loader, device)
            epoch_pbar.set_postfix({
                'train_loss': f"{train_loss/len(train_loader):.4f}",
                'val_loss': f"{val_loss:.4f}"
            })
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter > patience:
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        else:
            epoch_pbar.set_postfix({'train_loss': f"{train_loss/len(train_loader):.4f}"})

    epoch_pbar.close()

    # Restore best model if validation was used
    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_loss if use_early_stopping else loss.item()

def evaluate_model(model, data_loader, device=None):
    """Compute MSE on a given data loader."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * len(inputs)
    return total_loss / len(data_loader.dataset)

def run_inference_mse(model, inputs, outputs):
    """Compute MSE on the entire dataset (inputs, outputs)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(inputs_tensor)
    mse = nn.MSELoss()(predictions, outputs_tensor).item()
    return mse

# =========================================================
#   GMR: Fit Joint GMM and Perform Conditional Inference
# =========================================================

from sklearn.mixture import GaussianMixture

def fit_gmm_gmr(X, Y, n_components, covariance_type, reg_covar=1e-6,
                max_iter=500, n_init=3, random_state=SEED):
    """
    Fit a GMM on the joint data [X, Y].
    Returns the fitted GMM, plus the dimension sizes for X and Y.
    """
    n, x_dim = X.shape
    _, y_dim = Y.shape

    # Build joint data Z = [X, Y]
    Z = np.hstack([X, Y])

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state
    )

    gmm.fit(Z)  # Fit on joint space
    return gmm, x_dim, y_dim

def predict_gmr(gmm, X_in, x_dim, y_dim):
    """
    Perform Gaussian Mixture Regression:
      - For each mixture k, partition mean & covariance into:
         mu_k = [mu_k^x, mu_k^y], Sigma_k = [[Sigma_k^xx, Sigma_k^xy],
                                            [Sigma_k^yx, Sigma_k^yy]]
      - Compute posterior p(k|x_in).
      - Then y_hat_k(x_in) = mu_k^y + Sigma_k^yx * inv(Sigma_k^xx) * (x_in - mu_k^x)
      - Final y_hat = sum_k [ p(k|x_in) * y_hat_k(x_in) ].
    """
    means = gmm.means_
    covs = gmm.covariances_
    weights = gmm.weights_
    n_components = gmm.n_components

    N = X_in.shape[0]
    y_pred = np.zeros((N, y_dim))

    for i in range(N):
        x_i = X_in[i]

        numerators = np.zeros(n_components)
        y_component = np.zeros((n_components, y_dim))

        for k in range(n_components):
            mu_x = means[k, :x_dim]
            mu_y = means[k, x_dim:]

            # Partition covariance
            if gmm.covariance_type == 'full':
                Sigma = covs[k]
            elif gmm.covariance_type == 'diag':
                Sigma = np.diag(covs[k])
            elif gmm.covariance_type == 'tied':
                Sigma = gmm.covariances_
            elif gmm.covariance_type == 'spherical':
                Sigma = np.eye(x_dim + y_dim) * covs[k]
            else:
                raise NotImplementedError("Covariance type not implemented.")

            Sigma_xx = Sigma[:x_dim, :x_dim]
            Sigma_yx = Sigma[x_dim:, :x_dim]

            # Probability p(x_i | k)
            px = _gaussian_density(x_i, mu_x, Sigma_xx)
            numerators[k] = weights[k] * px

            # Conditional mean of y given x
            try:
                inv_Sigma_xx = np.linalg.inv(Sigma_xx)
            except np.linalg.LinAlgError:
                inv_Sigma_xx = np.linalg.pinv(Sigma_xx)
            diff_x = (x_i - mu_x).reshape(-1, 1)
            mu_y_given_x = mu_y.reshape(-1,1) + Sigma_yx @ inv_Sigma_xx @ diff_x
            y_component[k] = mu_y_given_x.ravel()

        denom = np.sum(numerators)
        if denom < 1e-15:
            alpha = np.ones(n_components) / n_components
        else:
            alpha = numerators / denom

        y_pred[i] = np.sum(alpha[:, None] * y_component, axis=0)

    return y_pred

def _gaussian_density(x, mean, cov):
    """
    Evaluate the multivariate normal pdf at x, with given mean and cov.
    """
    d = len(x)
    diff = x - mean
    try:
        cov_inv = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
        det_cov = np.linalg.det(cov_inv)
        if det_cov < 1e-15:
            return 0.0

    denom = np.sqrt((2*np.pi)**d * np.abs(det_cov)) + 1e-15
    exponent = -0.5 * diff @ cov_inv @ diff
    return np.exp(exponent) / denom

# =========================================================
#               Baselines: Linear Regression
# =========================================================

def run_linear_regression(inputs, outputs):
    """Fit LR on the entire dataset and compute MSE."""
    model = LinearRegression()
    model.fit(inputs, outputs)
    predictions = model.predict(inputs)
    mse_loss = np.mean((predictions - outputs) ** 2)
    return mse_loss

# =========================================================
#      Hyperparameter Tuning for GMM with GMR
# =========================================================

def hyperparameter_tuning_gmm_gmr(X_train, Y_train, X_val, Y_val, param_grid):
    """
    Grid search for GMM hyperparameters to do GMR on the training set,
    evaluate on validation set. Return best params + best val MSE.
    """
    best_params = None
    best_loss = float('inf')

    n_samples = X_train.shape[0]
    # Add progress bar for parameter grid search
    param_pbar = tqdm(ParameterGrid(param_grid), desc="GMM-GMR Grid Search", leave=False)
    
    for params in param_pbar:
        n_components = params['n_components']
        covariance_type = params['covariance_type']
        reg_covar = params['reg_covar']
        if n_components > n_samples:
            continue
        try:
            gmm, x_dim, y_dim = fit_gmm_gmr(
                X_train, Y_train,
                n_components=n_components,
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                random_state=SEED
            )
            Y_val_pred = predict_gmr(gmm, X_val, x_dim, y_dim)
            val_mse = np.mean((Y_val_pred - Y_val)**2)
            param_pbar.set_postfix({'val_mse': val_mse})

            if val_mse < best_loss:
                best_loss = val_mse
                best_params = params
        except ValueError as e:
            print(f"Skipping GMM with {params} due to error: {e}")
            continue

    param_pbar.close()
    return best_params, best_loss

def run_gmm_gmr_on_dataset(X_in, Y_in, best_params):
    """Fit GMM with best params on the entire (train+val) set and return the model, MSE."""
    gmm, x_dim, y_dim = fit_gmm_gmr(
        X_in, Y_in,
        n_components=best_params['n_components'],
        covariance_type=best_params['covariance_type'],
        reg_covar=best_params['reg_covar'],
        random_state=SEED
    )
    Y_pred = predict_gmr(gmm, X_in, x_dim, y_dim)
    mse = np.mean((Y_pred - Y_in) ** 2)
    return gmm, mse

# =========================================================
#                Main Runner
# =========================================================

def runner(cases, n_scenarios, nn_param_grid, gmm_param_grid):
    # We will store final results (test set performance)
    results_nn = []
    results_gmm = []
    results_lr = []

    # Add progress bar for case processing
    case_pbar = tqdm(cases, desc="Processing cases")
    
    for case in case_pbar:
        case_pbar.set_description(f"Processing {case}")
        print(f"\n>>>> Processing {case} with up to {n_scenarios} scenarios...\n")
        X_raw, Y_raw = generate_opf_dataset(case, n_scenarios=n_scenarios)

        if len(X_raw) < 30:
            # If fewer than 30 data points, not enough to do a meaningful train/val/test
            print(f"Not enough data for {case} (got {len(X_raw)}). Skipping.")
            continue

        # Scale data
        X_scaled, Y_scaled, in_scaler, out_scaler = scale_data(X_raw, Y_raw)

        # Train/Val/Test split
        X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(
            X_scaled, Y_scaled, test_size=0.15, val_size=0.15
        )
        print(f"Data Split => Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # 1) Neural Network Hyperparameter Tuning and Training
        print("\nTraining Neural Network...")
        best_nn_params, best_nn_val_loss = hyperparameter_tuning_nn(
            X_train, Y_train, X_val, Y_val, nn_param_grid
        )
        
        # Retrain best NN on Train+Val
        X_tv = np.vstack([X_train, X_val])
        Y_tv = np.vstack([Y_train, Y_val])

        ds_tv = OPFDataset(X_tv, Y_tv)
        loader_tv = DataLoader(ds_tv, batch_size=best_nn_params['batch_size'], shuffle=True)

        model_tv = NeuralNetwork(
            x_size=X_tv.shape[1],
            y_size=Y_tv.shape[1],
            hidden_sizes=best_nn_params['hidden_layer_config'],
            dropout_rate=best_nn_params['dropout_rate']
        )
        _ = train_model(
            model_tv, loader_tv, None,
            epochs=best_nn_params['epochs'],
            learning_rate=best_nn_params['learning_rate'],
            patience=3
        )
        
        # Evaluate on Test
        print("\nEvaluating Neural Network...")
        nn_test_mse = run_inference_mse(model_tv, X_test, Y_test)

        # 2) GMM-GMR Hyperparam Tuning on Train/Val
        print("\nTuning GMM-GMR...")
        best_gmm_params, best_gmm_val_loss = hyperparameter_tuning_gmm_gmr(
            X_train, Y_train, X_val, Y_val, gmm_param_grid
        )

        # Retrain best GMM on Train+Val
        print("\nTraining final GMM-GMR...")
        if best_gmm_params is not None:
            gmm_tv, gmm_tv_mse = run_gmm_gmr_on_dataset(X_tv, Y_tv, best_gmm_params)
            # Evaluate on Test
            gmm_xdim = X_tv.shape[1]
            gmm_ydim = Y_tv.shape[1]
            Y_test_pred = predict_gmr(gmm_tv, X_test, gmm_xdim, gmm_ydim)
            gmm_test_mse = np.mean((Y_test_pred - Y_test) ** 2)
        else:
            gmm_tv, gmm_tv_mse = None, np.nan
            gmm_test_mse = np.nan

        # 3) Linear Regression on Train+Val -> Test
        print("\nTraining and evaluating Linear Regression...")
        lr_model = LinearRegression().fit(X_tv, Y_tv)
        lr_test_preds = lr_model.predict(X_test)
        lr_test_mse = np.mean((lr_test_preds - Y_test)**2)

        # Store results
        results_nn.append({
            'Case': case,
            'Scenarios_final': len(X_raw),
            'Best NN Params': best_nn_params,
            'Val Loss': best_nn_val_loss,
            'Test MSE': nn_test_mse
        })
        results_gmm.append({
            'Case': case,
            'Scenarios_final': len(X_raw),
            'Best GMM Params': best_gmm_params,
            'Val Loss': best_gmm_val_loss,
            'Test MSE': gmm_test_mse
        })
        results_lr.append({
            'Case': case,
            'Scenarios_final': len(X_raw),
            'Test MSE': lr_test_mse
        })

    case_pbar.close()
    return results_nn, results_gmm, results_lr


# We update the NN hyperparam tuning to accept (X_train, Y_train, X_val, Y_val)
def hyperparameter_tuning_nn(X_train, Y_train, X_val, Y_val, param_grid):
    """Perform hyperparameter tuning for the neural network using the train/val sets."""
    best_params = None
    best_loss = float('inf')

    train_dataset = OPFDataset(X_train, Y_train)
    val_dataset   = OPFDataset(X_val,   Y_val)

    # Calculate total iterations for progress bar
    total_iters = len(list(ParameterGrid(param_grid)))
    pbar = tqdm(total=total_iters, desc="NN Hyperparameter Tuning", leave=False)

    for params in ParameterGrid(param_grid):
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=params['batch_size'], shuffle=False)

        # Create model
        model = NeuralNetwork(
            x_size=X_train.shape[1],
            y_size=Y_train.shape[1],
            hidden_sizes=params['hidden_layer_config'],
            dropout_rate=params['dropout_rate']
        )

        val_loss = train_model(
            model, train_loader, val_loader,
            epochs=params['epochs'],
            learning_rate=params['learning_rate'],
            patience=params.get('patience', 5)
        )
        
        pbar.update(1)
        pbar.set_postfix({'best_val_loss': best_loss, 'current_val_loss': val_loss})
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params

    pbar.close()
    return best_params, best_loss


def visualize_results(nn_res, gmm_res, lr_res):
    df_nn = pd.DataFrame(nn_res)
    df_gmm = pd.DataFrame(gmm_res)
    df_lr = pd.DataFrame(lr_res)

    print("\n============ Neural Network (Test Set) Results ============")
    print(df_nn)
    print("\n============ Gaussian Mixture Regression (Test Set) Results ============")
    print(df_gmm)
    print("\n============ Linear Regression (Test Set) Results ============")
    print(df_lr)


if __name__ == "__main__":
    # Cases of interest
    cases = ["case30", "case_ieee30", "case39", "case118", "GBreducednetwork"]

    # Number of random scenarios to generate for each network
    # Increase so that even after solver failures, we have enough data
    n_scenarios = 300

    # -------------------------------------------------
    # Neural Network Hyperparameter Grid
    # -------------------------------------------------
    nn_param_grid = {
        'batch_size': [16, 32],
        'learning_rate': [1e-3, 1e-4],
        'epochs': [100, 200],
        'hidden_layer_config': [
            [128, 128],
            [256, 256],
            [256, 256, 128]
        ],
        'dropout_rate': [0.0, 0.2],
        'patience': [10]
    }

    # -------------------------------------------------
    # GMM parameter grid (for GMR)
    # -------------------------------------------------
    gmm_param_grid = {
        'n_components': [2, 3, 5, 8],
        'covariance_type': ['full', 'diag'],
        'reg_covar': [1e-6, 1e-5]
    }

    nn_res, gmm_res, lr_res = runner(
        cases,
        n_scenarios,
        nn_param_grid,
        gmm_param_grid
    )

    visualize_results(nn_res, gmm_res, lr_res)