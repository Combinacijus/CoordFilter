import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob
import os

class GPSDataProcessor:
    def __init__(self, data_folder, output_folder, avg_vel_segment_len, vel_cutoff_offset):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.avg_vel_segment_len = avg_vel_segment_len
        self.vel_cutoff_offset = vel_cutoff_offset
        self.avg_vel = 0
        self.vel_cutoff = 0
        self.gps_data = []
        self.zoom_ax1_x = (None, None)
        self.zoom_ax1_y = (None, None)
        self.zoom_ax2_x = (None, None)
        self.zoom_ax2_y = (None, None)
        
        self.prefilter_avg_vel_mult = 5.0
        self.outliers_dict = {}  # Dictionary to hold different types of outliers

    # Load all GPS data files into a list of DataFrames
    def load_gps_data(self):
        file_paths = glob.glob(os.path.join(self.data_folder, "*.txt"))
        print(f"Files loaded: {file_paths}")
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            self.gps_data.append((os.path.basename(file_path), df))

    # Calculate velocity between consecutive points and average velocity over segment length
    def calculate_velocity(self, df):
        df['Time_Diff'] = df['Datetime'].diff().dt.total_seconds()
        df['Easting_Diff'] = df['Easting'].diff()
        df['Northing_Diff'] = df['Northing'].diff()
        df['Distance'] = np.sqrt(df['Easting_Diff']**2 + df['Northing_Diff']**2)
        df['Velocity'] = df['Distance'] / df['Time_Diff']
        df['Velocity'] = df['Velocity'] * 1.94384  # Convert m/s to knots
        df['Avg_Velocity'] = df['Velocity'].rolling(window=self.avg_vel_segment_len).mean()
        self.avg_vel = df['Velocity'].mean()
        self.vel_cutoff = self.avg_vel + self.vel_cutoff_offset
        return df
    
    # Prefilter data based on the velocity threshold and return both filtered and outlier data
    def prefilter_data(self, df):
        df['Velocity_Original'] = df['Velocity'].copy()
        
        prefilter_vel_high_cutoff = self.avg_vel * self.prefilter_avg_vel_mult
        prefilter_vel_low_cutoff = self.avg_vel / self.prefilter_avg_vel_mult
        
        mask = (
            (df['Velocity'] <= prefilter_vel_high_cutoff) & (df['Velocity'].shift(-1) <= prefilter_vel_high_cutoff)
            | (df['Velocity'] >= prefilter_vel_low_cutoff) & (df['Velocity'].shift(-1) >= prefilter_vel_low_cutoff)
        )
        
        df_filtered = df[mask]
        df_outlier_prefilter = df[~mask]
        
        # Save prefilter outliers in the dictionary
        self.outliers_dict['prefilter'] = df_outlier_prefilter
        print(self.outliers_dict)
        
        print(df_outlier_prefilter)
        
        return df_filtered, df_outlier_prefilter
    
    # Filter data based on the velocity threshold and combine outliers
    def filter_data(self, df):
        df = self.calculate_velocity(df)
        df, outliers_prefilter = self.prefilter_data(df)
        
        df = self.calculate_velocity(df)
        stats = self.calculate_statistics(df)
        self.print_statistics(stats)
            
        outlier_mask = (df['Velocity'] > self.vel_cutoff) & (df['Velocity'].shift(-1) > self.vel_cutoff)
        
        filtered_df = df[~outlier_mask]
        outliers_filter = df[outlier_mask]
        
        # Save final filter outliers in the dictionary
        self.outliers_dict['final_filter'] = outliers_filter
        
        combined_outliers = pd.concat([outliers_prefilter, outliers_filter])
        
        return filtered_df, combined_outliers

    # Calculate statistics like total distance, average velocity, max velocity, etc.
    def calculate_statistics(self, df):
        valid_velocities = df['Velocity'].dropna()
        valid_distances = df['Distance'].dropna()
        total_distance = valid_distances.sum()
        avg_vel = valid_velocities.mean()
        max_vel = valid_velocities.max()
        travel_heading = np.degrees(np.arctan2(df['Northing_Diff'].sum(), df['Easting_Diff'].sum()))
        total_time = df['Time_Diff'].sum()
        return {
            "Total Distance (m)": total_distance,
            "Average Velocity (knots)": avg_vel,
            "Maximum Velocity (knots)": max_vel,
            "Travel Heading (degrees)": travel_heading,
            "Total Time (s)": total_time
        }

    # Save debug data (including calculated columns) for debugging
    def save_debug_data(self, df, filename):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        output_path = os.path.join(self.output_folder, filename)
        df.to_csv(output_path, index=False, float_format="%.6f")

    # Plot data: trajectory and velocity graph including cutoff line and number of filtered points
    def plot_filtered_with_velocity(self, df, filtered_df, filename, ax1, ax2):
        ax1.clear()
        ax2.clear()
        
        # Plot original data and filtered points
        ax1.plot(df['Easting'], df['Northing'], marker='o', linestyle='-', label="Original Trajectory")
        
        # Plot each type of outlier in different colors and sizes
        colors = ['ro', 'go', 'bo']  # Define different colors for different outliers
        sizes = [10, 7, 4]  # Define different sizes for different outliers
        for idx, (outlier_type, outlier_df) in enumerate(self.outliers_dict.items()):
            if not outlier_df.empty:
                ax1.plot(outlier_df['Easting'], outlier_df['Northing'], colors[idx], markersize=sizes[idx], label=f"{outlier_type} Outliers ({len(outlier_df)} points)")
        
        ax1.set_title(f"Trajectory - {filename}")
        ax1.set_xlabel("Easting")
        ax1.set_ylabel("Northing")
        ax1.legend()
        
        # Plot velocity data with cutoff line and filtered points
        ax2.plot(df['Datetime'], df['Velocity'], 'b-', label="Velocity")
        
        for idx, (outlier_type, outlier_df) in enumerate(self.outliers_dict.items()):
            if not outlier_df.empty:
                ax2.plot(outlier_df['Datetime'], outlier_df['Velocity'], colors[idx], markersize=sizes[idx], label=f"{outlier_type} Outliers ({len(outlier_df)} points)")

        ax2.axhline(y=self.vel_cutoff, color='r', linestyle='--', label=f"Cutoff: {self.vel_cutoff:.2f} knots")
        ax2.axhline(y=self.avg_vel, color='k', linestyle=':', label=f"Avg Velocity: {self.avg_vel:.2f} knots", alpha=0.7)
        ax2.set_title(f"Velocity - {filename}")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Velocity (knots)")
        ax2.legend()

        plt.tight_layout()

        # Restore zoom
        if self.zoom_ax1_x != (None, None):
            ax1.set_xlim(self.zoom_ax1_x)
        if self.zoom_ax1_y != (None, None):
            ax1.set_ylim(self.zoom_ax1_y)
        if self.zoom_ax2_x != (None, None):
            ax2.set_xlim(self.zoom_ax2_x)
        if self.zoom_ax2_y != (None, None):
            ax2.set_ylim(self.zoom_ax2_y)

    # Interactive plot with slider to adjust velocity threshold
    def interactive_plot(self, df, filename):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        filtered_df, _ = self.filter_data(df)
        self.plot_filtered_with_velocity(df, filtered_df, filename, ax1, ax2)
        
        ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03])
        slider = Slider(ax_slider, 'Velocity Threshold', -10.0, 10.0, valinit=self.vel_cutoff_offset)

        # Store zoom levels and update plot
        def on_slider_update(val):
            self.vel_cutoff_offset = slider.val
            self.vel_cutoff = self.avg_vel + self.vel_cutoff_offset
            filtered_df, _ = self.filter_data(df)

            # Store zoom positions
            self.zoom_ax1_x = ax1.get_xlim()
            self.zoom_ax1_y = ax1.get_ylim()
            self.zoom_ax2_x = ax2.get_xlim()
            self.zoom_ax2_y = ax2.get_ylim()

            self.plot_filtered_with_velocity(df, filtered_df, filename, ax1, ax2)
            fig.canvas.draw_idle()

        slider.on_changed(on_slider_update)
        plt.show()

    # Print statistics
    def print_statistics(self, stats):
        print("Trajectory Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")

    # Main processing loop for all files
    def process_files(self):
        self.load_gps_data()
                
        for filename, df in self.gps_data:
            print(f"\nStatistics for {filename}:")
            
            self.save_debug_data(df, filename)
            self.interactive_plot(df, filename)

# Run the processing
if __name__ == "__main__":
    processor = GPSDataProcessor(data_folder="./data", output_folder="./data_debug", avg_vel_segment_len=20, vel_cutoff_offset=1)
    processor.process_files()
