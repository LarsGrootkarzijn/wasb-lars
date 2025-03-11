import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('/home/lars/WASB-SBDT/ball_with_flat_headers.csv')

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

# Define thresholds for pass detection
velocity_threshold = 5.0  # Example: Velocity threshold in pixels/sec
acceleration_threshold = 3000  # Example: Acceleration threshold in pixels/sec^2
change_threshold = 2.0  # Example: Change in direction (in pixels/sec)

# Initialize a list to store pass events
pass_events = []

# Loop through the data and detect passes
for i in range(1, len(df)):
    # Detect a pass when velocity and acceleration thresholds are met, with a change in direction
    if df['velocity'][i] > velocity_threshold and df['acceleration'][i] > acceleration_threshold:
        # Detect a significant change in direction or velocity (change from the previous point)
        if abs(df['Vx'][i] - df['Vx'][i-1]) > change_threshold or abs(df['Vy'][i] - df['Vy'][i-1]) > change_threshold:
            # Check if the previous point wasn't also part of a pass (avoid duplicate detection)
            if (i == 1) or (df['velocity'][i-1] <= velocity_threshold or df['acceleration'][i-1] <= acceleration_threshold):
                pass_events.append(df['time_seconds'][i])  # Store the time (in seconds) when a pass started

# Print detected pass events (time in seconds)
print(f"Pass events detected at times (seconds): {pass_events}")

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
plt.title('Velocity of Object with Detected Passes')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (pixels/sec)')
plt.grid(True)
plt.legend()

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig('velocity_acceleration_pass_start_plot.png')  # Save as 'velocity_acceleration_pass_start_plot.png'

# Optionally, show the plot
plt.show()
