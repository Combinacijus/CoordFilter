import numpy as np
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class SegmentFilter:
    def __init__(self, df, points_per_segment=300, offset=2):
        self.df = df
        self.points_per_segment = points_per_segment
        self.offset = offset
        self.total_points = len(df)
        self.num_segments = self.total_points // self.points_per_segment
        self.points_per_segment = self.total_points // self.num_segments

        # Initialize the RANSAC Regressor
        self.ransac = RANSACRegressor()

        # Initialize the lines list
        self.lines = []

        # Initialize the dashed lines lists
        self.dashed_lines_upper = []
        self.dashed_lines_lower = []

        # Initialize the scatter points lists
        self.scatter_points_inliers = []
        self.scatter_points_outliers = []

        # Initialize the text elements
        self.text_inliers = None
        self.text_outliers = None
        self.text_outliers_percentage = None

        # Create figure and axis for plotting
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)  # Adjust layout to make space for the slider

    def calculate_best_fit(self):
        for i in range(self.num_segments):
            # Define the start and end indices for this segment
            start_index = i * self.points_per_segment
            end_index = (i + 1) * self.points_per_segment if (i + 1) * self.points_per_segment < self.total_points else self.total_points

            df_segment = self.df.iloc[start_index:end_index]

            # Extract Easting and Northing data for this segment
            easting = df_segment['Easting'].values.reshape(-1, 1)  # Reshaped for sklearn
            northing = df_segment['Northing'].values

            # Use RANSAC instead of Linear Regression for robust fitting
            self.ransac.fit(easting, northing)
            slope = self.ransac.estimator_.coef_[0]
            intercept = self.ransac.estimator_.intercept_

            # Store slope and intercept for later use
            easting_range = np.linspace(easting.min(), easting.max(), 100)
            northing_fit = slope * easting_range + intercept

            # Store the line of best fit for the segment
            self.lines.append((slope, intercept, easting_range, northing_fit))

    def calculate_outlier_percentage(self, offset):
        total_inliers = 0
        total_outliers = 0

        for i in range(self.num_segments):
            slope, intercept, _, _ = self.lines[i]
            df_segment = self.df.iloc[i * self.points_per_segment:(i + 1) * self.points_per_segment]
            easting_segment = df_segment['Easting'].values
            northing_segment = df_segment['Northing'].values

            for j in range(len(df_segment)):
                predicted_northing = slope * easting_segment[j] + intercept
                distance_to_line = np.abs(northing_segment[j] - predicted_northing) / np.sqrt(1 + slope**2)

                if distance_to_line > offset:
                    total_outliers += 1
                else:
                    total_inliers += 1

        total_points = total_inliers + total_outliers
        return (total_outliers / total_points) * 100 if total_points > 0 else 0

    def plot_offset_vs_outlier_percentage(self, min_offset=0, max_offset=10, step=0.1):
        offsets = np.arange(min_offset, max_offset + step, step)
        outlier_percentages = []

        for offset in offsets:
            percentage = self.calculate_outlier_percentage(offset)
            outlier_percentages.append(percentage)
            
        # Plotting offset vs outlier percentage
        plt.figure(figsize=(10, 6))
        plt.plot(offsets, outlier_percentages, marker='o', color='b')
        plt.xlabel('Offset (m)')
        plt.ylabel('Outlier Percentage (%)')
        plt.title('Offset vs Outlier Percentage')
        plt.grid(True)
        plt.show()
        
    def initialize_plot(self):
        for i in range(self.num_segments):
            # Plot the line of best fit for the segment
            slope, intercept, easting_range, northing_fit = self.lines[i]
            self.ax.plot(easting_range, northing_fit, color='blue')

            # Initialize empty dashed lines and points (to be updated dynamically)
            dashed_upper, = self.ax.plot([], [], 'r--', alpha=0.4)
            dashed_lower, = self.ax.plot([], [], 'r--', alpha=0.4)
            self.dashed_lines_upper.append(dashed_upper)
            self.dashed_lines_lower.append(dashed_lower)

            # Plot the initial points and initialize empty scatter plots for inliers and outliers
            scatter_inliers = self.ax.scatter([], [], c='blue', alpha=0.6)
            scatter_outliers = self.ax.scatter([], [], c='red', marker='o')
            self.scatter_points_inliers.append(scatter_inliers)
            self.scatter_points_outliers.append(scatter_outliers)

    def update_dashed_lines(self):
        for i in range(self.num_segments):
            # Calculate the perpendicular offset points
            slope, intercept, easting_range, northing_fit = self.lines[i]
            perpendicular_dx = 1 / np.sqrt(1 + slope**2)
            perpendicular_dy = slope / np.sqrt(1 + slope**2)

            # Shift the best fit line by the perpendicular offset in both directions
            offset_easting_plus = easting_range + self.offset * perpendicular_dy
            offset_northing_plus = northing_fit - self.offset * perpendicular_dx
            offset_easting_minus = easting_range - self.offset * perpendicular_dy
            offset_northing_minus = northing_fit + self.offset * perpendicular_dx

            # Update the dashed lines without recreating them
            self.dashed_lines_upper[i].set_data(offset_easting_plus, offset_northing_plus)
            self.dashed_lines_lower[i].set_data(offset_easting_minus, offset_northing_minus)

            # Update the point colors and markers dynamically
            df_segment = self.df.iloc[i * self.points_per_segment:(i + 1) * self.points_per_segment]
            easting_segment = df_segment['Easting'].values
            northing_segment = df_segment['Northing'].values

            inliers_easting = []
            inliers_northing = []
            outliers_easting = []
            outliers_northing = []

            for j in range(len(df_segment)):
                # Calculate the closest point on the line of best fit
                predicted_northing = slope * easting_segment[j] + intercept

                # Calculate the perpendicular distance from the point to the line of best fit
                distance_to_line = np.abs(northing_segment[j] - predicted_northing) / np.sqrt(1 + slope**2)

                # If the distance is greater than the offset, mark as an outlier
                if distance_to_line > self.offset:
                    outliers_easting.append(easting_segment[j])
                    outliers_northing.append(northing_segment[j])
                else:
                    inliers_easting.append(easting_segment[j])
                    inliers_northing.append(northing_segment[j])

            # Update scatter plot data for inliers and outliers
            self.scatter_points_inliers[i].set_offsets(np.c_[inliers_easting, inliers_northing])
            self.scatter_points_outliers[i].set_offsets(np.c_[outliers_easting, outliers_northing])

    def update_text_elements(self):
        total_inliers = 0
        total_outliers = 0
        for i in range(self.num_segments):
            inliers_easting = self.scatter_points_inliers[i].get_offsets()[:, 0]
            outliers_easting = self.scatter_points_outliers[i].get_offsets()[:, 0]

            # Count total inliers and outliers
            total_inliers += len(inliers_easting)
            total_outliers += len(outliers_easting)

        # Update the number of inliers, outliers, and percentage of outliers
        if self.text_inliers is not None:
            self.text_inliers.remove()
        if self.text_outliers is not None:
            self.text_outliers.remove()
        if self.text_outliers_percentage is not None:
            self.text_outliers_percentage.remove()

        self.text_inliers = self.ax.text(0.05, 0.95, f"Total Inliers: {total_inliers}", transform=self.ax.transAxes, color='green')
        self.text_outliers = self.ax.text(0.05, 0.90, f"Total Outliers: {total_outliers}", transform=self.ax.transAxes, color='red')
        self.text_outliers_percentage = self.ax.text(0.05, 0.85, f"Percentage of Outliers: {total_outliers / (total_inliers + total_outliers) * 100:.2f}%", transform=self.ax.transAxes, color='blue')

    def plot(self):
        # Initialize plot with static elements
        self.initialize_plot()

        # Add a slider to control the offset value
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Position of the slider
        slider = Slider(ax_slider, 'Offset (m)', 0, 10.0, valinit=self.offset, valstep=0.01)

        # Update the plot when the slider is moved
        def update(val):
            self.offset = slider.val
            self.update_dashed_lines()
            self.update_text_elements()
            plt.draw()  # Redraw the figure

        slider.on_changed(update)

        # Show the plot
        plt.show()

# Example usage:
# Assuming df is your DataFrame containing 'Easting' and 'Northing' columns
# segment_filter = SegmentFilter(df, points_per_segment=300)
# segment_filter.calculate_best_fit()  # Calculate the best-fit lines
# segment_filter.plot_offset_vs_outlier_percentage(min_offset=0, max_offset=10, step=0.1)
