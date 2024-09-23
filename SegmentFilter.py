import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression

def calculate_best_fit_by_points_with_slider(df, points_per_segment=300, offset=2):
    total_points = len(df)
    num_segments = total_points // points_per_segment
    points_per_segment = total_points // num_segments

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)  # Adjust layout to make space for the slider

    # Store elements that need to be updated dynamically
    lines = []
    dashed_lines_upper = []
    dashed_lines_lower = []
    scatter_points = []

    def initialize_plot():
        for i in range(num_segments):
            # Define the start and end indices for this segment
            start_index = i * points_per_segment
            end_index = (i + 1) * points_per_segment if (i + 1) * points_per_segment < total_points else total_points

            df_segment = df.iloc[start_index:end_index]

            # Extract Easting and Northing data for this segment
            easting = df_segment['Easting'].values.reshape(-1, 1)  # Reshaped for sklearn
            northing = df_segment['Northing'].values

            # Perform linear regression on the segment
            reg = LinearRegression().fit(easting, northing)
            slope = reg.coef_[0]
            intercept = reg.intercept_

            # Store slope and intercept for later use
            easting_range = np.linspace(easting.min(), easting.max(), 100)
            northing_fit = slope * easting_range + intercept

            # Plot the line of best fit for the segment
            line, = ax.plot(easting_range, northing_fit, color='blue')
            lines.append((line, slope, intercept, easting_range, northing_fit))

            # Initialize empty dashed lines and points (to be updated dynamically)
            dashed_upper, = ax.plot([], [], 'r--')
            dashed_lower, = ax.plot([], [], 'r--')
            dashed_lines_upper.append(dashed_upper)
            dashed_lines_lower.append(dashed_lower)

            # Plot the initial points
            scatter = ax.scatter(df_segment['Easting'], df_segment['Northing'], alpha=0.6)
            scatter_points.append(scatter)

        # Set plot labels and title
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_title(f'Best Fit Lines with Perpendicular Offsets ({points_per_segment} points each)')

    def update_dashed_lines(offset):
        for i, (line, slope, intercept, easting_range, northing_fit) in enumerate(lines):
            # Calculate the perpendicular offset points
            perpendicular_dx = 1 / np.sqrt(1 + slope**2)
            perpendicular_dy = slope / np.sqrt(1 + slope**2)

            # Shift the best fit line by the perpendicular offset in both directions
            offset_easting_plus = easting_range + offset * perpendicular_dy
            offset_northing_plus = northing_fit - offset * perpendicular_dx
            offset_easting_minus = easting_range - offset * perpendicular_dy
            offset_northing_minus = northing_fit + offset * perpendicular_dx

            # Update the dashed lines without recreating them
            dashed_lines_upper[i].set_data(offset_easting_plus, offset_northing_plus)
            dashed_lines_lower[i].set_data(offset_easting_minus, offset_northing_minus)

            # Update the point colors dynamically
            df_segment = df.iloc[i * points_per_segment:(i + 1) * points_per_segment]
            easting_segment = df_segment['Easting'].values
            northing_segment = df_segment['Northing'].values

            outlier_mask = []
            for j in range(len(df_segment)):
                predicted_northing = slope * easting_segment[j] + intercept
                offset_upper = predicted_northing + offset * perpendicular_dx
                offset_lower = predicted_northing - offset * perpendicular_dx
                if northing_segment[j] > offset_upper or northing_segment[j] < offset_lower:
                    outlier_mask.append('red')
                else:
                    outlier_mask.append('blue')

            # Update scatter plot colors
            scatter_points[i].set_color(outlier_mask)

    # Initialize plot with static elements
    initialize_plot()

    # Add a slider to control the offset value
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Position of the slider
    slider = Slider(ax_slider, 'Offset (m)', 0, 10.0, valinit=offset, valstep=0.01)

    # Update the plot when the slider is moved
    def update(val):
        offset = slider.val
        update_dashed_lines(offset)
        fig.canvas.draw_idle()  # Redraw the figure

    slider.on_changed(update)

    plt.show()

# Example usage:
# calculate_best_fit_by_points_with_slider(df, points_per_segment=300)
