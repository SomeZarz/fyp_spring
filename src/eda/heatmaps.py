import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# === Parse waveform columns ===
def robust_parse_waveform_column(col):
    parsed = []
    for val in col:
        if isinstance(val, (int, float)):
            parsed.append(val)
        elif isinstance(val, str) and '~' in val:
            try:
                nums = [float(x.strip()) for x in val.split('~') if x.strip()]
                parsed.append(np.mean(nums) if nums else np.nan)
            except:
                parsed.append(np.nan)
        else:
            parsed.append(np.nan)
    return pd.Series(parsed)

def strg_corr(df):
    df['terminaltime'] = pd.to_timedelta(df['terminaltime'], unit='s')

    # fixing datetime
    reference_time = pd.Timestamp("2024-01-01")
    df['terminaltime'] = reference_time + df['terminaltime']
    df['batteryvoltage_clean'] = robust_parse_waveform_column(df['batteryvoltage'])
    df['probetemperatures_clean'] = robust_parse_waveform_column(df['probetemperatures'])

    # === Handle Missing Data ===
    df.dropna(subset=['batteryvoltage_clean', 'probetemperatures_clean'], inplace=True)

    # === Correlation Analysis ===
    corr_matrix = df.corr(numeric_only=True)
    soc_corr = corr_matrix['soc'].drop('soc')

    # Features with strong correlation to SOC
    strong_corr = soc_corr[abs(soc_corr) > 0.4].sort_values(ascending=False)
    print("Features with |correlation| > 0.4 to SOC:")
    print(strong_corr)
    return strong_corr

def heatmap(df, strong_corr):
    # Heatmap of strong correlations
    strong_features = strong_corr.index.tolist() + ['soc']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[strong_features].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap (SOC and Strong Correlates)')
    plt.tight_layout()
    plt.show()

    # Barplot of strong correlation features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=strong_corr.values, y=strong_corr.index, palette='coolwarm', orient='h')
    plt.title('Features with |Correlation| > 0.4 with SOC')
    plt.xlabel('Correlation Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Time Series Plots ===
    plt.figure(figsize=(12, 6))
    plt.plot(df['terminaltime'], df['totalvoltage'], color='orange', label='Total Voltage')
    plt.title('Total Voltage vs Time')
    plt.xlabel('Time')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df['terminaltime'], df['totalcurrent'], color='green', label='Total Current')
    plt.title('Total Current vs Time')
    plt.xlabel('Time')
    plt.ylabel('Current (A)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df['terminaltime'], df['mintemperaturevalue'], label='Min Temp', color='blue')
    plt.plot(df['terminaltime'], df['maxtemperaturevalue'], label='Max Temp', color='red')
    plt.title('Battery Temperature Range Over Time')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Scatter Plots Against SOC ===
    plt.figure(figsize=(10, 6))
    plt.scatter(df['soc'], df['totalvoltage'], alpha=0.3, s=10)
    plt.xlabel('SOC')
    plt.ylabel('Total Voltage (V)')
    plt.title('Voltage vs SOC')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['soc'], df['batteryvoltage_clean'], alpha=0.3, s=10, color='purple')
    plt.xlabel('SOC')
    plt.ylabel('Battery Voltage (Avg)')
    plt.title('Battery Voltage vs SOC')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['soc'], df['probetemperatures_clean'], alpha=0.3, s=10, color='darkred')
    plt.xlabel('SOC')
    plt.ylabel('Probe Temperature (Avg)')
    plt.title('Probe Temperature vs SOC')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

'''import os

output_folder = "correct_heatmaps_for_input"
os.makedirs(output_folder, exist_ok=True)

# Raw time x feature maps
for i in range(0, len(df_scaled) - window_size, step_size):
    window_data = df_scaled[heatmap_features].iloc[i:i+window_size]

    # Save heatmap image
    plt.figure(figsize=(8, 6))
    sns.heatmap(window_data.T, cmap='viridis', cbar=True)
    plt.title(f"Raw Time-Feature Heatmap {i}")
    plt.xlabel('Time Steps')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/heatmap_raw_{i}.png")
    plt.close()

    # Save as NumPy array
    np.save(f"{output_folder}/heatmap_raw_{i}.npy", window_data.T.values)'''

