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
velocity_threshold = 1000  # Example threshold in pixels per second
df_filtered = df[df["Velocity"] < velocity_threshold]

# Alternatively, using IQR (Interquartile Range) to filter out outliers
Q1 = df["Velocity"].quantile(0.25)
Q3 = df["Velocity"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_filtered = df_filtered[(df_filtered["Velocity"] >= lower_bound) & (df_filtered["Velocity"] <= upper_bound)]
df_filtered["Frame"] = df_filtered["Frame"] / 25

# --- Apply Moving Average (Last 10 Frames) ---
df_filtered["Velocity_MA"] = df_filtered["Velocity"].ewm(span=10, adjust=False).mean()

# Detect Passes
def detect_passes(df):
    passes = []
    last_pass_frame = None  # Store the frame of the last detected pass to avoid duplicates
    for i in range(1, len(df)):
        # Pass detection criteria:
        if (df["Velocity"].iloc[i] >= 50) and \
           (df["Displacement"].iloc[i] >= 10) and \
           (df["Time_Diff"].iloc[i] <= 0.05):  # Adjust criteria as needed
            # Ensure this is the first frame of the pass (avoid duplicate lines)
            if last_pass_frame is None or df["Frame"].iloc[i] > last_pass_frame + 1:
                passes.append(df.iloc[i])
                last_pass_frame = df["Frame"].iloc[i]  # Update the last detected pass frame
    
    return pd.DataFrame(passes)

# Detect passes based on the criteria
df["Frame"] = df["Frame"] / 25
passes_df = detect_passes(df[:500])

# Detect Passes Received
def detect_passes_received(df, passes_df):
    received_passes = []
    reception_threshold = 50  # Velocity drop threshold (adjustable)
    distance_threshold = 5  # Low distance threshold for pass reception (adjustable)
    
    for pass_frame in passes_df["Frame"]:
        # Find the index of the detected pass
        pass_idx = df[df["Frame"] == pass_frame].index[0]
        
        # Check subsequent frames for velocity drop and small displacement (indicating reception)
        for i in range(pass_idx + 1, len(df)):
            if df["Velocity"].iloc[i] > reception_threshold and df["Displacement"].iloc[i] < distance_threshold:
                received_passes.append(df.iloc[i])
                break  # Stop after detecting the first reception event
    
    return pd.DataFrame(received_passes)

# Detect passes received based on the ball slowing down significantly after a pass
df_received_passes = detect_passes_received(df[:500], passes_df[:500])

# Plot the filtered velocity data with moving average
plt.figure(figsize=(20, 5))
plt.plot(df_filtered["Frame"][:500], df_filtered["Velocity"][:500], label="Velocity (pixels/s)", color="r", alpha=0.5)
plt.plot(df_filtered["Frame"][:500], df_filtered["Velocity_MA"][:500], label="Velocity (pixels/s)", color="y", alpha=0.5)
# Highlight the detected passes with vertical lines
for pass_frame in passes_df["Frame"]:
    plt.axvline(x=pass_frame, color='blue', linestyle='--', label="Pass Event", linewidth=2)

# Highlight the detected pass receptions with vertical lines
for received_frame in df_received_passes["Frame"]:
    plt.axvline(x=received_frame, color='green', linestyle='--', label="Pass Received", linewidth=2)

# Add labels and title
plt.xlabel("Frame Number")
plt.ylabel("Velocity (pixels/s)")
plt.title("Football Velocity Over Time (Cleaned, Outliers Removed, and Moving Average Applied)")

# Add x-ticks for better visibility
#x_ticks = np.arange(df_filtered["Frame"].min(), df_filtered["Frame"].max(), step=4)
#plt.xticks(x_ticks)

# Show the legend and grid
plt.grid()

# Save the plot
plt.savefig("velocity_plot_with_passes.png", dpi=300)  # High-resolution save
plt.show()

print("Plot saved as velocity_plot_with_passes.png")
