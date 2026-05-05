import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# --- FIX RANDOMNESS ---
np.random.seed(2)
# %% Systems' Parameters
n = 3
m = 2
p = 3

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

D = np.zeros((p, m))

# --- EXECUTION ---
L, N, Nt = 2, 30, 10000

# Define Distinct Horizons
L_out = L          # Output Horizon
L_in  = L      # Input Horizon (L + n)

# Define Noise Statistics
mean = 1
sigma  = 2

# Define grid boundaries
m1_grid = np.linspace(0, 1.5, 200)
m2_grid = np.linspace(2.5, 7, 200)
# %% Condition Checks
print("--- Checking Dimensions ---")
rows_H = (m * L_in) + (p * L_out) 
cols_H = (N - L_in + 1)

print(f"Parameters: N={N}, L_out={L_out}, L_in={L_in}, Nt={Nt}")
print(f"Hankel Block Rows: {rows_H}")
print(f"Hankel Block Cols: {cols_H}")

if rows_H < cols_H:
    print("Result: Rows < Cols. Individual Matrix has more columns than rows.")
else:
    print("Result: Rows >= Cols. Individual Matrix has more rows than columns.")
print("---------------------------\n")

# %% Hankel Matrices Generation

def generate_noisy_hankel_data(L_in, L_out, N, Nt, A, B, C, D):
    n = A.shape[0]
    p = C.shape[0]
    m = B.shape[1]
    
    list_HU, list_HY = [], []
    
    # The number of columns is constrained by the longest horizon (L_in)
    n_cols = N - L_in + 1
    cols_h = [f'Col_{j}' for j in range(n_cols)]

    # Initialize accumulators for the squared noisy Hankel matrices
    HUsq = np.zeros((m * L_in, n_cols))
    HYsq = np.zeros((p * L_out, n_cols))
    HUYCross = np.zeros((m * L_in+p * L_out, m * L_in+p * L_out))
    HUYC = np.zeros((m * L_in+p * L_out, m * L_in+p * L_out))
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
            # A. Fill Clean Matrices
            u_flat = u[:, k : k + L_in].flatten(order='F')
            HU_clean[:, k] = u_flat 
            
            y_flat = y[:, k : k + L_out].flatten(order='F')
            HY_clean[:, k] = y_flat 
            
            # B. Fill Noise Matrices
            u_noise_flat = u_noise_seq[:, k : k + L_in].flatten(order='F')
            HU_noise_mat[:, k] = u_noise_flat
            
            y_noise_flat = y_noise_seq[:, k : k + L_out].flatten(order='F')
            HY_noise_mat[:, k] = y_noise_flat
        
        HUYClean = np.vstack((HU_clean,HY_clean))
        
        
        # 4. Final Noisy Hankel Matrices
        HU_final = HU_clean + HU_noise_mat
        HY_final = HY_clean + HY_noise_mat
        
        HUY = np.vstack((HU_final, HY_final))
        
        HUYCross += HUY @ HUY.T
        HUYC += HUYClean @ HUYClean.T 
        
        # --- NEW: Accumulate Element-wise Squares of NOISY Data ---
        HUsq += HU_final**2
        HYsq += HY_final**2

        # Convert to DataFrame
        df_u = pd.DataFrame(HU_final, columns=cols_h)
        df_y = pd.DataFrame(HY_final, columns=cols_h)
        df_u.insert(0, 'ID', i)
        df_y.insert(0, 'ID', i)
        list_HU.append(df_u)
        list_HY.append(df_y)
    
    # Compute Average of Squares
    HUYC = HUYC / Nt
    HUsq = HUsq / Nt
    HYsq = HYsq / Nt
    HUYCross = HUYCross / Nt
    return pd.concat(list_HU), pd.concat(list_HY), HUsq, HYsq, HUYCross, HUYC
#start_time = time.perf_counter()
# Generate Data
HU_data, HY_data, HUsq, HYsq, HUYCross, HUYC = generate_noisy_hankel_data(L_in, L_out, N, Nt, A, B, C, D)
#end_time = time.perf_counter()
SVTRUE1 = np.linalg.svd(HUYC, compute_uv=False)
print("SVTRUE1 (Singular Values of uncorrected HUYCross) calculated.")

# --- Setup 2D Grid Search ---
print("\n--- Starting 2D Grid Search (m1 and m2) ---")

# Define grid boundaries


# True parameters for reference
true_m1 = mean
true_m2 = sigma**2 + mean**2

# Pre-calculate expected sums for the bias matrix
E_HU = HU_data.drop(columns=['ID']).groupby(level=0).mean().values
E_HY = HY_data.drop(columns=['ID']).groupby(level=0).mean().values
E_HUY = np.vstack((E_HU, E_HY))
S = np.sum(E_HUY, axis=1)
S_col = S.reshape(-1, 1)
S_row = S.reshape(1, -1)
rows_M, cols_M = E_HUY.shape

# --- DYNAMIC THRESHOLD CALCULATION ---
# Calculate the theoretically perfect matrix first to find the real noise floor
B_true = (true_m1 * S_col) + (true_m1 * S_row) - (cols_M * (true_m1**2))
B_true += np.eye(rows_M) * (cols_M * (true_m2 - true_m1**2))
M_true = HUYCross.copy() - B_true
SVTRUE = np.linalg.svd(M_true, compute_uv=False)

# Expected Rank = m * L_in + n = (2 * 2) + 3 = 7
expected_rank = m * L_in + n

# Set threshold safely in the "gap" between the 7th and 8th singular values
# (Index 6 and 7 in 0-indexed Python)
rank_threshold = (SVTRUE[expected_rank - 1] + SVTRUE[expected_rank]) / 2.0
print(f"Theoretical System Rank: {expected_rank}")
print(f"Dynamically set Rank Threshold: {rank_threshold:.4f}")

# Initialize matrices
min_sv_matrix = np.zeros((len(m1_grid), len(m2_grid)))
rank_matrix = np.zeros((len(m1_grid), len(m2_grid))) 
start_time = time.perf_counter()
# Execute Grid Search
for i, curr_m1 in enumerate(m1_grid):
    for j, curr_m2 in enumerate(m2_grid):
        
        # Mathematical constraint: Variance must be non-negative
        if curr_m2 < curr_m1**2:
            min_sv_matrix[i, j] = np.nan
            rank_matrix[i, j] = np.nan 
            continue
            
        curr_var = curr_m2 - curr_m1**2
        B = (curr_m1 * S_col) + (curr_m1 * S_row) - (cols_M * (curr_m1**2))
        B += np.eye(rows_M) * (cols_M * curr_var)
        
        M_corrected = HUYCross.copy() - B
        s_values = np.linalg.svd(M_corrected, compute_uv=False)
        
        min_sv_matrix[i, j] = np.min(s_values)
        
        # Estimate rank using our dynamically found threshold
        rank_matrix[i, j] = np.sum(s_values > rank_threshold)

end_time = time.perf_counter()
# --- FIND MINIMUM SINGULAR VALUE FROM GRID SEARCH ---
# Find the flattened index of the absolute minimum, ignoring NaNs
flat_min_idx = np.nanargmin(min_sv_matrix)

# Convert the 1D flattened index back into 2D grid coordinates (i, j)
i_min, j_min = np.unravel_index(flat_min_idx, min_sv_matrix.shape)

# --- FIND MINIMUM SINGULAR VALUE FROM GRID SEARCH ---
# Find the flattened index of the absolute minimum, ignoring NaNs
flat_min_idx = np.nanargmin(min_sv_matrix)

# Convert the 1D flattened index back into 2D grid coordinates (i, j)
i_min, j_min = np.unravel_index(flat_min_idx, min_sv_matrix.shape)

# Extract the corresponding m1, m2, and the exact singular value
est_m1 = m1_grid[i_min]
est_m2 = m2_grid[j_min]
min_sv_val = min_sv_matrix[i_min, j_min]

print("\n--- Minimum Singular Value in Grid Search ---")
print(f"Estimated m1: {est_m1:.4f}")
print(f"Estimated m2: {est_m2:.4f}")
print(f"Min SV Value: {min_sv_val:.6f}")
print("---------------------------------------------\n")

# --- Plotting: Heatmap for Estimated Rank ---
plt.figure(figsize=(10, 8))

M1, M2 = np.meshgrid(m1_grid, m2_grid)

# Define discrete color boundaries for integer ranks
min_rank = np.nanmin(rank_matrix)
max_rank = np.nanmax(rank_matrix)

# Use a colormap that makes the "crater" visually distinct
cmap = plt.get_cmap('magma', int(max_rank - min_rank + 1))
pm = plt.pcolormesh(M1, M2, rank_matrix.T, cmap=cmap, shading='nearest', vmin=min_rank-0.5, vmax=max_rank+0.5)

# Setup colorbar to show exact integer ticks
cbar = plt.colorbar(pm, ticks=np.arange(min_rank, max_rank + 1))
cbar.set_label(f'Estimated Rank', rotation=270, labelpad=20)

# Plot true parameters
plt.plot(true_m1, true_m2, marker='*', color='cyan', markersize=30, markeredgecolor='black', label=f'True (m1={true_m1}, m2={true_m2})')

plt.title('Heatmap: Matrix Rank vs. Moments')
plt.xlabel('Assumed Mean ($m_1$)')
plt.ylabel('Assumed Second Moment ($m_2$)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# --- Plotting: SVTRUE (To verify the gap) ---
plt.figure(figsize=(10, 6))
plt.stem(SVTRUE)
plt.axhline(y=rank_threshold, color='r', linestyle='--', label=f'Dynamic Threshold')
plt.title(f'Singular Values of Bias-Corrected Matrix at True Parameters')
plt.xlabel('Index')
plt.ylabel('Singular Value Magnitude')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# --- Plotting: Heatmap for Minimum Singular Value ---
plt.figure(figsize=(10, 8))

# 1. Plot the continuous landscape of the minimum singular values
cp = plt.contourf(M1, M2, min_sv_matrix.T, levels=50, cmap='viridis')
cbar = plt.colorbar(cp)
cbar.set_label('Minimum Singular Value ($\sigma_{min}$)', rotation=270, labelpad=20)

# 2. Highlight the specific 0.01 threshold boundary
# This draws a dashed white line exactly where the minimum singular value crosses 0.01
plt.contour(M1, M2, min_sv_matrix.T, levels=[0.001], colors='white', linewidths=2, linestyles='dashed')

# 3. Plot the true parameters
plt.plot(true_m1, true_m2, marker='*', color='cyan', markersize=30, markeredgecolor='black', label=f'True ($m_1$={true_m1}, $m_2$={true_m2})')

plt.title('Heatmap: Minimum Singular Value vs. Moments\n(White dashed line indicates threshold $\sigma_{min} < 0.01$)')
plt.xlabel('Assumed Mean ($m_1$)')
plt.ylabel('Assumed Second Moment ($m_2$)')
plt.grid(True, alpha=0.3)

# Add a custom legend entry for the contour line
import matplotlib.lines as mlines
contour_line = mlines.Line2D([], [], color='white', linestyle='--', linewidth=2, label='$\sigma_{min} = 0.001$ Boundary')
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(contour_line)
plt.legend(handles=handles, loc='upper right')

plt.show()

# --- EXPORT DATA TO CSV FOR MATLAB ---
print("\n--- Exporting Data to CSV ---")

export_data = []
for i, m1_val in enumerate(m1_grid):
    for j, m2_val in enumerate(m2_grid):
        # Extract the calculated values (including np.nan)
        rank_val = rank_matrix[i, j]
        sv_val = min_sv_matrix[i, j]
        export_data.append([m1_val, m2_val, rank_val, sv_val])

# Create a DataFrame and save to CSV
df_export = pd.DataFrame(export_data, columns=['m1', 'm2', 'rank', 'min_sv'])
df_export.to_csv('heatmap_data.csv', index=False)

print("Saved successfully to 'heatmap_data.csv'.")
execution_time = end_time - start_time
print(f"The averaging and SVD steps took {execution_time:.4f} seconds.")