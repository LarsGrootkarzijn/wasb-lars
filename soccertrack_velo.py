import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# File paths
csv_file_path = '/home/lars/WASB-SBDT/ball_with_flat_headers.csv'  # Replace with your actual CSV path
video_file_path = '/home/lars/Downloads/archive(3)/top_view/videos/D_20220220_1_0000_0030.mp4'  # Replace with the actual video path

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Assuming the same steps from before (calculating velocity and acceleration)
frame_rate = 30
df['dx'] = df['bb_left'].diff()  # Difference in bb_left (horizontal movement)
df['dy'] = df['bb_top'].diff()   # Difference in bb_top (vertical movement)
df['time_seconds'] = df['frame'] / frame_rate  # Convert frame to time in seconds
df['Vx'] = df['dx'] / (1 / frame_rate)  # Horizontal velocity (pixels/sec)
df['Vy'] = df['dy'] / (1 / frame_rate)  # Vertical velocity (pixels/sec)
df['velocity'] = np.sqrt(df['Vx']**2 + df['Vy']**2)
df['dV'] = df['velocity'].diff()  # Change in velocity
df['dt'] = df['time_seconds'].diff()  # Time difference
df['acceleration'] = df['dV'] / df['dt']  # Calculate acceleration

# Define thresholds for pass detection and deflection detection
velocity_threshold = 300.0  # Example: Velocity threshold in pixels/sec
acceleration_threshold = 3000  # Example: Acceleration threshold in pixels/sec^2
change_threshold = 0.0  # Example: Change in direction (in pixels/sec)
deflection_velocity_threshold = -300  # Large negative velocity change indicating a deflection
deflection_acceleration_threshold = -4000  # Large negative acceleration change indicating a deflection
deflection_duration = 3  # How long the negative velocity change must persist to be considered a deflection
angle_max_change_threshold = 180.0  # Maximum allowed angle change (degrees)
angle_min_change_threshold = 25.0  # Minimum allowed angle change (degrees)
min_time_gap = 0  # 1 second
# Initialize lists to store pass events and deflections
pass_events = []
deflection_events = []
pass_frames = []  # List to store corresponding frame numbers
deflection_frames = []  # List to store deflection frame numbers

# Loop through the data and detect passes and deflections
for i in range(1, len(df)):
    # Detect large negative spikes in velocity or acceleration (potential deflection)
    if df['velocity'][i] < deflection_velocity_threshold or df['acceleration'][i] < deflection_acceleration_threshold:
        # Check if the negative spike is significant (indicating a deflection)
        deflection_events.append(df['time_seconds'][i])  # Store the time when a deflection was detected
        deflection_frames.append(df['frame'][i])  # Store the corresponding frame number
        continue  # Skip this iteration as it's a deflection, not a pass
    
    # Detect a pass (significant positive change in velocity or acceleration)
    if df['velocity'][i] > velocity_threshold and df['acceleration'][i] > acceleration_threshold:
        # Detect a significant change in direction or velocity (change from the previous point)
        if abs(df['Vx'][i] - df['Vx'][i-1]) > change_threshold or abs(df['Vy'][i] - df['Vy'][i-1]) > change_threshold:
            # Check if the previous point wasn't also part of a pass (avoid duplicate detection)
            if (i == 1) or (df['velocity'][i-1] <= velocity_threshold or df['acceleration'][i-1] <= acceleration_threshold):
                pass_events.append(df['time_seconds'][i])  # Store the time (in seconds) when a pass started
                pass_frames.append(df['frame'][i])  # Store the corresponding frame number

# Calculate the angles of the trajectory
df['angle'] = np.arctan2(df['dy'], df['dx']) * (180 / np.pi)  # Convert from radians to degrees

# Calculate the angle change between consecutive frames
df['angle_change'] = df['angle'].diff().abs()

# Define threshold for significant angle change (e.g., 40 degrees)

# Initialize a list to store frames with significant angle change
angle_change_frames = df[df['angle_change'] > angle_min_change_threshold]

# Set the minimum time gap for direction change (in seconds)

# Filter angle_change_frames to remove outliers based on the time difference
filtered_angle_change_frames = []

# Initialize the first frame
last_angle_change_time = None

for index, row in angle_change_frames.iterrows():
    current_angle_change_time = row['time_seconds']
    
    # If it's the first angle change or the time difference from the last is more than 1 second
    if last_angle_change_time is None or (current_angle_change_time - last_angle_change_time) >= min_time_gap:
        filtered_angle_change_frames.append(row)
        last_angle_change_time = current_angle_change_time  # Update the last angle change time

# Convert the filtered results back to a DataFrame for annotation
filtered_angle_change_frames_df = pd.DataFrame(filtered_angle_change_frames)

# Filter angle change frames using angle thresholds (min and max angle change)
filtered_angle_change_frames_df = filtered_angle_change_frames_df[
    (filtered_angle_change_frames_df['angle_change'] >= angle_min_change_threshold) & 
    (filtered_angle_change_frames_df['angle_change'] <= angle_max_change_threshold)
]

# Print detected pass events (time in seconds) and corresponding frames
print(f"Pass events detected at times (seconds): {pass_events}")
print(f"Corresponding frames for passes: {pass_frames}")
print(f"Deflection events detected at times (seconds): {deflection_events}")
print(f"Corresponding frames for deflections: {deflection_frames}")
print(f"Frames with filtered angle changes (>= 1 second gap and within angle thresholds):")
print(filtered_angle_change_frames_df[['frame', 'angle', 'angle_change']])

# Plotting the velocity, acceleration, and detected passes
plt.figure(figsize=(14, 8))

# Plot velocity
plt.subplot(3, 1, 1)
plt.plot(df['time_seconds'], df['velocity'], color='b', label="Velocity (pixels/sec)")
plt.title('Velocity of Object Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (pixels/sec)')
plt.grid(True)
plt.legend()

# Plot acceleration
plt.subplot(3, 1, 2)
plt.plot(df['time_seconds'], df['acceleration'], color='r', label="Acceleration (pixels/sec^2)")
plt.title('Acceleration of Object Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (pixels/sec^2)')
plt.grid(True)
plt.legend()

# Plot passes on velocity graph
plt.subplot(3, 1, 3)
plt.plot(df['time_seconds'], df['velocity'], color='b', label="Velocity (pixels/sec)")
plt.scatter(pass_events, [df.loc[df['time_seconds'] == time, 'velocity'].values[0] for time in pass_events],
            color='g', label="Detected Passes", zorder=5)  # Plot only detected passes as green dots
plt.scatter(deflection_events, [df.loc[df['time_seconds'] == time, 'velocity'].values[0] for time in deflection_events],
            color='r', label="Deflections", zorder=5)  # Plot deflections as red dots
plt.title('Velocity of Object with Detected Passes and Deflections')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (pixels/sec)')
plt.grid(True)
plt.legend()

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig('velocity_acceleration_pass_deflection_plot.png')  # Save as 'velocity_acceleration_pass_deflection_plot.png'

# Optionally, show the plot
plt.show()

# OpenCV: Extract frames from video for detected passes
# Initialize OpenCV video capture object
cap = cv2.VideoCapture(video_file_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video frame rate (fps) and total frame count
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the directory to save extracted frames
output_dir = './extracted_frames/'
os.makedirs(output_dir, exist_ok=True)

# Loop through the pass events (frame numbers) and extract those frames
for pass_frame in pass_frames:
    if pass_frame <= frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pass_frame)  # Set the video to the desired frame
        ret, frame = cap.read()  # Read the frame
        if ret:
            # Save the frame as an image
            output_file = os.path.join(output_dir, f"pass_frame_{pass_frame}.png")
            cv2.imwrite(output_file, frame)
            print(f"Frame {pass_frame} extracted and saved to {output_file}")
        else:
            print(f"Error: Could not read frame {pass_frame}")
    else:
        print(f"Warning: Frame {pass_frame} exceeds video frame count")

# Release the video capture object
cap.release()

# **Adding the full trajectory plot here:**
plt.figure(figsize=(10, 6))

# Flip Y axis if needed depending on your coordinate system
plt.gca().invert_yaxis()  # Optional, depending on how your Y data behaves

# Plot the full trajectory of the ball
plt.plot(df['bb_left'], df['bb_top'], color='blue', linewidth=2, label='Ball trajectory')

# Plot pass events
pass_x = df.loc[df['frame'].isin(pass_frames), 'bb_left']
pass_y = df.loc[df['frame'].isin(pass_frames), 'bb_top']
plt.scatter(pass_x, pass_y, color='green', s=100, label='Pass events', zorder=5)

# Plot deflection events
deflection_x = df.loc[df['frame'].isin(deflection_frames), 'bb_left']
deflection_y = df.loc[df['frame'].isin(deflection_frames), 'bb_top']
plt.scatter(deflection_x, deflection_y, color='red', s=100, label='Deflection events', zorder=5)

# Add arrows to show direction (optional, for some key frames)
for i in range(1, len(df), 10):  # Every 10 frames for clarity
    plt.arrow(df['bb_left'][i-1], df['bb_top'][i-1],
              df['bb_left'][i] - df['bb_left'][i-1],
              df['bb_top'][i] - df['bb_top'][i-1],
              head_width=5, head_length=5, fc='gray', ec='gray', alpha=0.5)

# Plot start and end points
plt.scatter(df['bb_left'].iloc[0], df['bb_top'].iloc[0], color='yellow', s=150, marker='*', label='Start', zorder=5)
plt.scatter(df['bb_left'].iloc[-1], df['bb_top'].iloc[-1], color='black', s=150, marker='X', label='End', zorder=5)

# Field layout (example dimensions)
plt.title('Ball Trajectory with Angle Change')
plt.xlabel('X Position (pixels)')
plt.ylabel('Y Position (pixels)')
plt.legend()
plt.grid(True)

# Optionally, set axis limits based on video resolution or field size
plt.xlim(0, df['bb_left'].max() + 50)
plt.ylim(0, df['bb_top'].max() + 50)

# Annotate significant angle changes
for index, row in filtered_angle_change_frames_df.iterrows():
    plt.annotate(f"{row['angle_change']:.2f}°", 
                 (df['bb_left'][row.name], df['bb_top'][row.name]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='purple')

plt.tight_layout()
plt.savefig('ball_trajectory_angle_changes.png')
plt.show()
