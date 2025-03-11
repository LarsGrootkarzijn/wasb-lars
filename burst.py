import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the XML file
file_path = "/home/lars/WASB-SBDT/datasets/soccer/annos/ID-3.xml"  # Update with the correct path if necessary
tree = ET.parse(file_path)
root = tree.getroot()

# Extract frame numbers and positions of the football
data = []
for track in root.findall(".//track[@label='ball']"):
    for point in track.findall("points"):
        frame = int(point.get("frame"))
        coords = point.get("points").split(",")
        x, y = float(coords[0]), float(coords[1])
        data.append([frame, x, y])

# Convert to pandas DataFrame
df = pd.DataFrame(data, columns=["Frame", "X", "Y"])

# Sort DataFrame by frame number
df = df.sort_values(by="Frame").reset_index(drop=True)

# Remove rows where X or Y is NaN (if necessary)
df = df.dropna(subset=["X", "Y"])

# Compute displacement vectors (difference in positions)
df["DX"] = df["X"].diff()
df["DY"] = df["Y"].diff()

# Compute frame differences & time differences
df["Frame_Diff"] = df["Frame"].diff()
frame_time = 1 / 25  # Assuming 30 FPS, change to your frame rate if different
df["Time_Diff"] = df["Frame_Diff"] * frame_time

# Remove NaN values (to avoid issues in calculations)
df = df.dropna(subset=["Time_Diff"])

# Compute displacement (distance traveled)
df["Displacement"] = (df["DX"]**2 + df["DY"]**2) ** 0.5

# Compute velocity (pixels per second)
df["Velocity"] = df["Displacement"] / df["Time_Diff"]

# --- Remove Outliers ---
velocity_threshold = 10000000  # Example threshold in pixels per second
df_filtered = df[df["Velocity"] < velocity_threshold]

# Alternatively, using IQR (Interquartile Range) to filter out outliers
#Q1 = df["Velocity"].quantile(0.25)
#Q3 = df["Velocity"].quantile(0.75)
#IQR = Q3 - Q1
#lower_bound = Q1 - 1.5 * IQR
#upper_bound = Q3 + 1.5 * IQR

#df_filtered = df_filtered[(df_filtered["Velocity"] >= lower_bound) & (df_filtered["Velocity"] <= upper_bound)]
df_filtered["Frame"] = df_filtered["Frame"] / 25

# --- Apply Moving Average (Last 10 Frames) ---
df_filtered["Velocity_MA"] = df_filtered["Velocity"].ewm(span=10, adjust=False).mean()

# --- Burst Detection --- 
# Calculate acceleration (change in velocity over time)
df_filtered["Acceleration"] = df_filtered["Velocity"].diff()
df_filtered["acc_EMA"] = df_filtered["Acceleration"].ewm(span=10, adjust=False).mean()

from scipy.signal import savgol_filter

#df_filtered['acc_EMA'] = savgol_filter(df_filtered['Acceleration'], window_length=11, polyorder=6)


# Define burst detection criteria
burst_threshold = 200  # Define a threshold for burst in acceleration (change as needed)
burst_duration = 5    # Minimum number of frames for a burst to be considered valid

# Detect bursts: when acceleration exceeds the burst threshold
df_filtered["Burst"] = df_filtered["Acceleration"] > burst_threshold

# Identify start of bursts where velocity continues to rise after the initial burst
burst_starts = []
start_frame = None

for i in range(1, len(df_filtered)):
    if df_filtered["Burst"].iloc[i] and not df_filtered["Burst"].iloc[i - 1]:
        # This is the start of a burst, check if velocity continues to rise
        start_frame = df_filtered["Frame"].iloc[i]

        # Check subsequent frames for continued velocity increase
        if i + 1 < len(df_filtered) and df_filtered["Velocity"].iloc[i + 1] > df_filtered["Velocity"].iloc[i]:
            # If velocity continues to increase, add this as a valid burst start
            burst_starts.append(start_frame)

# Plot the filtered velocity data with moving average
plt.figure(figsize=(20, 5))
plt.plot(df_filtered["Frame"][600:], df_filtered["Acceleration"][600:], label="Velocity (pixels/s)", color="r", alpha=0.5)

# Plot only the start of each burst with vertical dashed lines
#for start in burst_starts:
#    plt.axvline(x=start, color='blue', linestyle='--', linewidth=2)
x_ticks = np.arange(70, df_filtered["Frame"].max(), step=1)  # Set step to 1 second
plt.xticks(x_ticks)
# Add labels and title
plt.xlabel("Frame Number")
plt.ylabel("Velocity (pixels/s)")
plt.title("Football Velocity Over Time with Burst Detection")

# Show the legend and grid
plt.legend()
plt.grid()

# Save the plot
plt.savefig("velocity_plot_with_bursts_start_only_filtered.png", dpi=300)  # High-resolution save
plt.show()

print("Plot saved as velocity_plot_with_bursts_start_only_filtered.png")
