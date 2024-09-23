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

    # Create the initial plot without the offset
    lines = []  # Store the lines for the best fit and dashed lines
    def plot_lines(offset=offset):
        ax.clear()  # Clear the plot for updates
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

            # Plot the points in the segment
            ax.scatter(easting, northing, alpha=0.6)

            # Plot the line of best fit for the segment
            easting_range = np.linspace(easting.min(), easting.max(), 100)
            northing_fit = slope * easting_range + intercept
            line, = ax.plot(easting_range, northing_fit, color='blue')

            # Calculate the perpendicular offset points
            perpendicular_dx = 1 / np.sqrt(1 + slope**2)
            perpendicular_dy = slope / np.sqrt(1 + slope**2)

            # Shift the best fit line by the perpendicular offset in both directions
            offset_easting_plus = easting_range + offset * perpendicular_dy
            offset_northing_plus = northing_fit - offset * perpendicular_dx
            offset_easting_minus = easting_range - offset * perpendicular_dy
            offset_northing_minus = northing_fit + offset * perpendicular_dx

            # Plot the dashed perpendicular lines at the offset
            ax.plot(offset_easting_plus, offset_northing_plus, 'r--')
            ax.plot(offset_easting_minus, offset_northing_minus, 'r--')

            # Find points outside the offset lines
            for idx in range(len(df_segment)):
                point_easting = df_segment.iloc[idx]['Easting']
                point_northing = df_segment.iloc[idx]['Northing']
                predicted_northing = slope * point_easting + intercept

                # Calculate the offset lines at the point's easting
                offset_upper = predicted_northing + offset * perpendicular_dx
                offset_lower = predicted_northing - offset * perpendicular_dx

                # Check if the point is outside the offset bounds
                if point_northing > offset_upper or point_northing < offset_lower:
                    # Plot the point in red with 'X' marker
                    ax.scatter(point_easting, point_northing, color='red', marker='x', s=100)
                else:
                    # Plot the point normally
                    ax.scatter(point_easting, point_northing, color='blue', marker='o', s=50)

            lines.append(line)

        # Set plot labels and title
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_title(f'Best Fit Lines with Perpendicular Offsets ({points_per_segment} points each)')
        # ax.legend()

    # Plot the initial lines with no offset
    plot_lines()

    # Add a slider to control the offset value
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Position of the slider
    slider = Slider(ax_slider, 'Offset (m)', 0, 10.0, valinit=offset, valstep=0.01)

    # Update the plot when the slider is moved
    def update(val):
        offset = slider.val
        plot_lines(offset)
        fig.canvas.draw_idle()  # Redraw the figure

    slider.on_changed(update)

    plt.show()
