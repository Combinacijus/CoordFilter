import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob
import os

class GPSDataProcessor:
    def __init__(self, data_folder, output_folder, avg_vel_segment_len, vel_cutoff_offset):
        self.data_folder = data_folder
        self.path_out_filtered = output_folder
        self.path_out_debug = output_folder + "_debug"
        self.avg_vel_segment_len = avg_vel_segment_len
        self.vel_cutoff_offset = vel_cutoff_offset
        self.avg_vel = 0
        self.vel_cutoff = 0
        self.gps_data_list = []
        self.zoom_ax1_x = (None, None)
        self.zoom_ax1_y = (None, None)
        self.zoom_ax2_x = (None, None)
        self.zoom_ax2_y = (None, None)
        self.zoom_ax3_x = (None, None)
        self.zoom_ax3_y = (None, None)
        self.zoom_ax4_x = (None, None)
        self.zoom_ax4_y = (None, None)
        
        self.prefilter_avg_vel_mult = 5.0
        self.outliers_dict = {}  # Dictionary to hold different types of outliers

    # Load all GPS data files into a list of DataFrames and save list of columns
    def load_gps_data(self):
        file_paths = glob.glob(os.path.join(self.data_folder, "*.txt"))
        print(f"Files loaded:\n{file_paths}")
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()  # Save list of columns
            self.gps_data_list.append((os.path.basename(file_path), df, columns))
            
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            
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
        df = self.calculate_velocity(df)
        df['Velocity_Original'] = df['Velocity'].copy()
        
        prefilter_vel_high_cutoff = self.avg_vel * self.prefilter_avg_vel_mult
        prefilter_vel_low_cutoff = self.avg_vel / self.prefilter_avg_vel_mult
        
        outlier_mask = (
            (df['Velocity'] > prefilter_vel_high_cutoff) & (df['Velocity'].shift(-1) > prefilter_vel_high_cutoff)
            | (df['Velocity'] < prefilter_vel_low_cutoff) & (df['Velocity'].shift(-1) < prefilter_vel_low_cutoff)
        )
        
        df_prefiltered = df[~outlier_mask]
        df_outlier_prefilter = df[outlier_mask]
        
        # Save prefilter outliers in the dictionary
        self.outliers_dict['prefilter'] = df_outlier_prefilter
        print(self.outliers_dict)
        
        print(df_outlier_prefilter)
        
        self.df_prefiltered = df_prefiltered
        
        return df_prefiltered, df_outlier_prefilter
    
    # Filter data based on the velocity threshold and combine outliers
    def filter_data(self, df):
        df_prefiltered, outliers_prefilter = self.prefilter_data(df)
        df = df_prefiltered.copy()
        
        df = self.calculate_velocity(df)
        stats = self.calculate_statistics(df)
        self.print_statistics(stats)
            
        outlier_mask = (df['Velocity'] > self.vel_cutoff) & (df['Velocity'].shift(-1) > self.vel_cutoff)
        df_filtered = df[~outlier_mask]
        outliers_filter = df[outlier_mask]
        
        # Save final filter outliers in the dictionary
        self.outliers_dict['filter'] = outliers_filter
        
        combined_outliers = pd.concat([outliers_prefilter, outliers_filter])
        
        self.df_filtered = df_filtered
        
        return df_filtered, combined_outliers

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
        
    def save_data(self, df, folder, filename, columns=None):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        output_path = os.path.join(folder, filename)
        if columns is not None:
            df[columns].to_csv(output_path, index=False, float_format="%.4f")
        else:
            df.to_csv(output_path, index=False, float_format="%.4f")
            

    # Plot data: trajectory and velocity graph including cutoff line and number of filtered points
    def plot_all(self, df, df_filtered, filename, ax1, ax2, ax3, ax4):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        
        # Plot original data and filtered points
        ax1.plot(df['Easting'], df['Northing'], marker='o', linestyle='-', label="Original Trajectory")
        
        colors_rgba = [(1, 0, 0, 1), (1, 0.5, 0, 1), (1, 1, 0, 1)]  # RGBA
        sizes = [30, 20, 10]
        for idx, (outlier_type, outlier_df) in enumerate(self.outliers_dict.items()):
            if not outlier_df.empty:
                ax1.scatter(outlier_df['Easting'], outlier_df['Northing'], color=colors_rgba[idx], s=sizes[idx], label=f"{outlier_type} Outliers ({len(outlier_df)} points)", zorder=10)
        
        ax1.set_title(f"Trajectory - {filename}")
        ax1.set_xlabel("Easting")
        ax1.set_ylabel("Northing")
        ax1.legend()
        
        # Plot velocity data with cutoff line and filtered points
        ax2.plot(df['Datetime'], df['Velocity'], 'b-', label="Velocity")
        
        for idx, (outlier_type, outlier_df) in enumerate(self.outliers_dict.items()):
            if not outlier_df.empty:
                ax2.scatter(outlier_df['Datetime'], outlier_df['Velocity'], color=colors_rgba[idx], s=sizes[idx], label=f"{outlier_type} Outliers ({len(outlier_df)} points)", zorder=10)

        ax2.axhline(y=self.vel_cutoff, color='r', linestyle='--', label=f"Cutoff: {self.vel_cutoff:.2f} knots")
        ax2.axhline(y=self.avg_vel, color='k', linestyle=':', label=f"Avg Velocity: {self.avg_vel:.2f} knots", alpha=0.7)
        ax2.set_title(f"Velocity - {filename}")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Velocity (knots)")
        ax2.legend()

        # Plot data after prefilter
        # ax3.plot(df['Easting'], df['Northing'], marker='o', linestyle='-', label="Original Trajectory")
        # ax3.scatter(self.df_prefiltered['Easting'], self.df_prefiltered['Northing'], color='b', s=10, label="Prefiltered Data", zorder=10)
        ax3.plot(self.df_prefiltered['Easting'], self.df_prefiltered['Northing'], marker='o', linestyle='-', color='b', label="Prefiltered Data", zorder=10)
        ax3.set_title(f"Prefiltered Trajectory - {filename}")
        ax3.set_xlabel("Easting")
        ax3.set_ylabel("Northing")
        ax3.legend()

        # Plot data after all filters
        # ax4.plot(df['Easting'], df['Northing'], marker='o', linestyle='-', label="Original Trajectory")
        ax4.plot(df_filtered['Easting'], df_filtered['Northing'], marker='o', linestyle='-', color='b', label="Filtered Data", zorder=10)
        # ax4.scatter(df_filtered['Easting'], df_filtered['Northing'], color='b', s=10, label="Filtered Data", zorder=10)
        ax4.set_title(f"Filtered Trajectory - {filename}")
        ax4.set_xlabel("Easting")
        ax4.set_ylabel("Northing")
        ax4.legend()

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
        if self.zoom_ax3_x != (None, None):
            ax3.set_xlim(self.zoom_ax3_x)
        if self.zoom_ax3_y != (None, None):
            ax3.set_ylim(self.zoom_ax3_y)
        if self.zoom_ax4_x != (None, None):
            ax4.set_xlim(self.zoom_ax4_x)
        if self.zoom_ax4_y != (None, None):
            ax4.set_ylim(self.zoom_ax4_y)

    # Interactive plot with slider to adjust velocity threshold
    def interactive_plot_and_filtering(self, df, filename):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
        df_filtered, _ = self.filter_data(df)
        self.plot_all(df, df_filtered, filename, ax1, ax2, ax3, ax4)
        
        ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03])
        slider = Slider(ax_slider, 'Velocity Threshold', -10.0, 10.0, valinit=self.vel_cutoff_offset)

        # Store zoom levels and update plot
        def on_slider_update(val):
            self.vel_cutoff_offset = slider.val
            self.vel_cutoff = self.avg_vel + self.vel_cutoff_offset
            df_filtered, _ = self.filter_data(df)

            # Store zoom positions
            self.zoom_ax1_x = ax1.get_xlim()
            self.zoom_ax1_y = ax1.get_ylim()
            self.zoom_ax2_x = ax2.get_xlim()
            self.zoom_ax2_y = ax2.get_ylim()
            self.zoom_ax3_x = ax3.get_xlim()
            self.zoom_ax3_y = ax3.get_ylim()
            self.zoom_ax4_x = ax4.get_xlim()
            self.zoom_ax4_y = ax4.get_ylim()

            self.plot_all(df, df_filtered, filename, ax1, ax2, ax3, ax4)
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
        
        for filename, df, columns in self.gps_data_list:
            print(f"\nWorking on a file:   {filename}")
            
            self.interactive_plot_and_filtering(df, filename)
            
            # TODO
            # self.save_data(df, self.path_out_debug, filename)
            # self.save_data(df, self.path_out_filtered, filename, columns)

# Run the processing
if __name__ == "__main__":
    processor = GPSDataProcessor(data_folder="./data", output_folder="./data_filtered", avg_vel_segment_len=20, vel_cutoff_offset=1)
    processor.process_files()
