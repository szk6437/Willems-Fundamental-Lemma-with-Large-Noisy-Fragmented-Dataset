import time
from typing import Tuple
import numpy as np
import pandas as pd

def initialize_system_matrices() -> Tuple[np.ndarray, ...]:
    """Defines the discrete-time state-space system matrices (A, B, C, D)."""
    A = np.array([
        [0.8, -0.1, 0.0],
        [0.1,  0.7, 0.1],
        [0.0, -0.2, 0.6]
    ])
    
    B = np.array([
        [1.0, 0.0],  
        [0.0, 1.0],  
        [0.5, 0.5]   
    ])
    
    C = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    p, m = C.shape[0], B.shape[1]
    D = np.zeros((p, m))
    
    return A, B, C, D

def generate_noisy_hankel_data(
    L_in: int, L_out: int, N: int, Nt: int, 
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
    mean: float, sigma: float
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates clean and noisy Hankel matrices from system trajectories."""
    n = A.shape[0]
    p = C.shape[0]
    m = B.shape[1]
    
    list_HU, list_HY = [], []
    n_cols = N - L_in + 1
    cols_h = [f'Col_{j}' for j in range(n_cols)]

    HUsq = np.zeros((m * L_in, n_cols))
    HYsq = np.zeros((p * L_out, n_cols))
    HUYCross = np.zeros((m * L_in + p * L_out, m * L_in + p * L_out))
    HUYC = np.zeros((m * L_in + p * L_out, m * L_in + p * L_out))
    
    for i in range(Nt):
        x0 = np.random.randn(n) 
        if i % 5000 == 0:
            print(f"Processing iteration {i}/{Nt}...")
            
        # 1. Generate Clean Trajectory
        u = np.random.randn(m, N)
        x = x0.copy()
        y = np.zeros((p, N))
        
        for k in range(N):
            y[:, k] = C @ x + D @ u[:, k]
            x = A @ x + B @ u[:, k]
        
        # 2. Generate Noise Trajectories
        u_noise_seq = np.random.normal(mean, sigma, (m, N))
        y_noise_seq = np.random.normal(mean, sigma, (p, N))
        
        # 3. Construct Hankel Matrices
        HU_clean = np.zeros((m * L_in, n_cols))
        HY_clean = np.zeros((p * L_out, n_cols))
        HU_noise_mat = np.zeros((m * L_in, n_cols))
        HY_noise_mat = np.zeros((p * L_out, n_cols))
        
        for k in range(n_cols):
            HU_clean[:, k] = u[:, k : k + L_in].flatten(order='F')
            HY_clean[:, k] = y[:, k : k + L_out].flatten(order='F')
            HU_noise_mat[:, k] = u_noise_seq[:, k : k + L_in].flatten(order='F')
            HY_noise_mat[:, k] = y_noise_seq[:, k : k + L_out].flatten(order='F')
        
        HUYClean = np.vstack((HU_clean, HY_clean))
        
        # 4. Final Noisy Hankel Matrices
        HU_final = HU_clean + HU_noise_mat
        HY_final = HY_clean + HY_noise_mat
        HUY = np.vstack((HU_final, HY_final))
        
        # Matrix accumulations
        HUYCross += HUY @ HUY.T
        HUYC += HUYClean @ HUYClean.T 
        HUsq += HU_final**2
        HYsq += HY_final**2

        # Convert to DataFrame
        df_u = pd.DataFrame(HU_final, columns=cols_h)
        df_y = pd.DataFrame(HY_final, columns=cols_h)
        df_u.insert(0, 'ID', i)
        df_y.insert(0, 'ID', i)
        list_HU.append(df_u)
        list_HY.append(df_y)
    
    # Compute Expected Values
    HUYC /= Nt
    HUsq /= Nt
    HYsq /= Nt
    HUYCross /= Nt
    
    return pd.concat(list_HU), pd.concat(list_HY), HUsq, HYsq, HUYCross, HUYC

def compute_dynamic_threshold(
    HUYCross: np.ndarray, S_col: np.ndarray, S_row: np.ndarray, 
    rows_M: int, cols_M: int, true_m1: float, true_m2: float, expected_rank: int
) -> float:
    """Calculates the theoretical matrix gap to establish an optimal SVD rank threshold."""
    B_true = (true_m1 * S_col) + (true_m1 * S_row) - (cols_M * (true_m1**2))
    B_true += np.eye(rows_M) * (cols_M * (true_m2 - true_m1**2))
    M_true = HUYCross.copy() - B_true
    
    SVTRUE = np.linalg.svd(M_true, compute_uv=False)
    
    # Set threshold safely in the "gap"
    rank_threshold = (SVTRUE[expected_rank - 1] + SVTRUE[expected_rank]) / 2.0
    return rank_threshold

def main():
    # --- Configuration & Setup ---
    np.random.seed(2)
    A, B, C, D = initialize_system_matrices()
    n, p, m = A.shape[0], C.shape[0], B.shape[1]
    
    L, N, Nt = 2, 30, 10000
    L_out = L          
    L_in  = L          
    
    mean_noise = 1
    sigma_noise = 2
    true_m1 = mean_noise
    true_m2 = sigma_noise**2 + mean_noise**2
    
    # --- Matrix Dimension Checks ---
    rows_H = (m * L_in) + (p * L_out) 
    cols_H = (N - L_in + 1)
    
    print("--- System Constraints ---")
    print(f"Parameters: N={N}, L_out={L_out}, L_in={L_in}, Nt={Nt}")
    print(f"Hankel Block Rows: {rows_H} | Hankel Block Cols: {cols_H}")
    print("--------------------------\n")
    
    # --- Data Generation ---
    print("--- Generating Trajectory Data ---")
    HU_data, HY_data, HUsq, HYsq, HUYCross, HUYC = generate_noisy_hankel_data(
        L_in, L_out, N, Nt, A, B, C, D, mean_noise, sigma_noise
    )
    
    # --- Expected Sums Calculation ---
    E_HU = HU_data.drop(columns=['ID']).groupby(level=0).mean().values
    E_HY = HY_data.drop(columns=['ID']).groupby(level=0).mean().values
    E_HUY = np.vstack((E_HU, E_HY))
    
    S = np.sum(E_HUY, axis=1)
    S_col = S.reshape(-1, 1)
    S_row = S.reshape(1, -1)
    rows_M, cols_M = E_HUY.shape

    # --- Rank & Threshold Mapping ---
    expected_rank = m * L_in + n
    rank_threshold = compute_dynamic_threshold(
        HUYCross, S_col, S_row, rows_M, cols_M, true_m1, true_m2, expected_rank
    )
    
    # --- Grid Search Initialization ---
    m1_grid = np.linspace(0, 1.5, 200)
    m2_grid = np.linspace(2.5, 7, 200)
    min_sv_matrix = np.zeros((len(m1_grid), len(m2_grid)))
    rank_matrix = np.zeros((len(m1_grid), len(m2_grid))) 
    
    print("--- Executing Grid Search ---")
    start_time = time.perf_counter()
    
    for i, curr_m1 in enumerate(m1_grid):
        for j, curr_m2 in enumerate(m2_grid):
            # Mathematical constraint: Variance must be non-negative
            if curr_m2 < curr_m1**2:
                min_sv_matrix[i, j] = np.nan
                rank_matrix[i, j] = np.nan 
                continue
                
            curr_var = curr_m2 - curr_m1**2
            B_mat = (curr_m1 * S_col) + (curr_m1 * S_row) - (cols_M * (curr_m1**2))
            B_mat += np.eye(rows_M) * (cols_M * curr_var)
            
            M_corrected = HUYCross.copy() - B_mat
            s_values = np.linalg.svd(M_corrected, compute_uv=False)
            
            min_sv_matrix[i, j] = np.min(s_values)
            rank_matrix[i, j] = np.sum(s_values > rank_threshold)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # --- Extract Results ---
    flat_min_idx = np.nanargmin(min_sv_matrix)
    i_min, j_min = np.unravel_index(flat_min_idx, min_sv_matrix.shape)
    
    est_m1 = m1_grid[i_min]
    est_m2 = m2_grid[j_min]
    min_sv_val = min_sv_matrix[i_min, j_min]
    
    # --- Left Null Space Calculation ---
    # Re-evaluate the corrected matrix at the optimal estimated moments
    optimal_var = est_m2 - est_m1**2
    B_optimal = (est_m1 * S_col) + (est_m1 * S_row) - (cols_M * (est_m1**2))
    B_optimal += np.eye(rows_M) * (cols_M * optimal_var)
    M_optimal = HUYCross.copy() - B_optimal
    
    U, S_vals, Vh = np.linalg.svd(M_optimal, full_matrices=True)
    # The basis of the left null space corresponds to the left singular vectors
    # associated with the singular values beyond the theoretical rank.
    left_null_space_basis = U[:, expected_rank:]

    # --- Final Output ---
    print("\n" + "="*40)
    print(" GRID SEARCH RESULTS")
    print("="*40)
    print(f"Grid Search Runtime : {execution_time:.4f} seconds")
    print(f"Expected Rank       : {expected_rank}")
    print(f"Dynamically Set Tol : {rank_threshold:.4f}")
    print("-" * 40)
    print(" ESTIMATED NOISE MOMENTS")
    print("-" * 40)
    print(f"True m1 (Mean)      : {true_m1:.4f}  | Est m1: {est_m1:.4f}")
    print(f"True m2 (2nd Moment): {true_m2:.4f}  | Est m2: {est_m2:.4f}")
    print(f"Minimum SV Achieved : {min_sv_val:.6f}")
    print("-" * 40)
    print(" LEFT NULL SPACE BASIS (U[:, expected_rank:])")
    print("-" * 40)
    print(f"Dimensions          : {left_null_space_basis.shape}")
    print("Basis Vectors (First 3 rows truncated for display):")
    print(np.round(left_null_space_basis[:3, :], 4))
    print("...")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
