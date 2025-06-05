import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def info(df):
    # drop 'Unnamed: 0'(index) column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    print("Basic Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

    ##missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # visualize the distribution of SOC
    plt.figure(figsize=(8, 5))
    sns.histplot(df['soc'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of State of Charge (SOC)')
    plt.xlabel('SOC')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def corr(df):
    corr_matrix = df.corr(numeric_only=True)
    soc_corr = corr_matrix['soc'].drop('soc').sort_values(ascending=False)

    # features most positively correlated with SOC
    print("\nTop 5 positively correlated features with SOC:")
    print(soc_corr.head(5))

    # features most negatively correlated with SOC
    print("\nTop 5 negatively correlated features with SOC:")
    print(soc_corr.tail(5))

    # Heatmap of SOC and top correlated features
    top_corr_features = soc_corr.abs().sort_values(ascending=False).head(10).index.tolist()
    top_corr_features.append('soc')
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap (Top SOC-related Features)')
    plt.show()
    return corr_matrix


def feature_eng(df, corr_matrix):
    soc_corr = corr_matrix['soc'].drop('soc')

    # Filter features with correlation > 0.4
    strong_corr = soc_corr[abs(soc_corr) > 0.4]

    print("Features with |correlation| > 0.4 with SOC:")
    print(strong_corr.sort_values(ascending=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=strong_corr.values, y=strong_corr.index, palette='coolwarm', orient='h')
    plt.title('Features with |Correlation| > 0.4 with SOC')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.grid(True)
    plt.show()

    #heatmap of SOC and Strong Correlates
    strong_corr_features = strong_corr.index.tolist() + ['soc']
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[strong_corr_features].corr(), annot=True, cmap='RdBu_r', center=0)
    plt.title('Correlation Heatmap (SOC and Strong Correlates)')
    plt.show()

    ##dropped columns
    all_features = soc_corr.index.tolist()

    dropped_columns = list(set(all_features) - set(strong_corr_features))

    print(f"Dropped features (|correlation with SOC| ≤ 0.4):\n{dropped_columns}")

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