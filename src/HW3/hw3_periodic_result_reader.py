import time
import matplotlib.pyplot as plt
import numpy as np
import os

reward_list_path = "HW3/rewards_list_2024.12.03-16:58:18.pth"

# Create subplots: one for raw data, one for smoothed data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Initialize the line objects for raw and smoothed data
line_raw, = ax1.plot([], [], label="Raw Training Reward", color="blue")
ax1.set_title("Raw Training Reward")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Reward")
ax1.legend()

line_smoothed, = ax2.plot([], [], label="Smoothed Training Reward", color="red")
ax2.set_title("Smoothed Training Reward")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Reward")
ax2.legend()


# Function to smooth data using an exponential moving average
def smooth(data, alpha=0.1):
    smoothed = []
    current = data[0]
    for value in data:
        current = alpha * value + (1 - alpha) * current
        smoothed.append(current)
    return np.array(smoothed)


# Update the plots
def update_plots(data, line_raw, line_smoothed, ax1, ax2):
    # Update the raw data plot
    line_raw.set_xdata(range(len(data)))
    line_raw.set_ydata(data)
    ax1.relim()
    ax1.autoscale_view()

    # Smooth the data with a larger window size or exponential smoothing
    smoothed_data = smooth(data, alpha=0.01)

    # Update the smoothed data plot
    line_smoothed.set_xdata(range(len(smoothed_data)))
    line_smoothed.set_ydata(smoothed_data)
    ax2.relim()
    ax2.autoscale_view()

    # Refresh the plots
    plt.draw()
    fig.canvas.flush_events()


while True:
    try:
        # Read the data from file
        data = []
        with open(reward_list_path, "r") as f:
            for ln in f:
                data.append(float(ln.strip()))  # Assuming each line is a single float value
        data = np.array(data)

        print(f"Read {len(data)} data points from {reward_list_path}")

        # Update plots
        update_plots(data, line_raw, line_smoothed, ax1, ax2)

        # Save the updated figure as PNG
        plt.savefig("updated_plot.png", bbox_inches='tight')

        # Flush the file explicitly to ensure immediate writing
        with open("updated_plot.png", "rb") as png_file:
            os.fsync(png_file.fileno())

    except Exception as e:
        print(f"Error reading or plotting data: {e}")

    # Pause for 60 seconds before the next update
    time.sleep(5)