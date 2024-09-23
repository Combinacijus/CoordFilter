import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob
import os
from sklearn.linear_model import LinearRegression


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
        self.zoom_ax_x = (None, None)
        self.zoom_ax1_y = (None, None)
        self.zoom_ax2_y = (None, None)
        self.zoom_ax3_y = (None, None)
        self.zoom_ax4_y = (None, None)

        self.prefilter_avg_vel_mult = 5.0  # If vel is bigger or smaller by this value then delete in prefilter

        self.df_original = None
        self.df_prefiltered = None
        self.df_filtered = None

        self.outliers_dict = {}  # Dictionary to hold different types of outliers
        # self.outliers_prefilter = None
        # self.outliers_filter = None

    # Load all GPS data files into a list of DataFrames and save list of columns
    def load_gps_data(self):
        file_paths = glob.glob(os.path.join(self.data_folder, "*.txt"))
        print(f"Files loaded:\n{file_paths}")

        for file_path in file_paths:
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()  # Save list of columns
            self.gps_data_list.append((os.path.basename(file_path), df, columns))

            df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

    # Calculate velocity between consecutive points and average velocity over segment length
    def df_append_calc_vel(self, df):
        df = df.copy()
        df["Time_Diff"] = df["Datetime"].diff().dt.total_seconds()
        df["Easting_Diff"] = df["Easting"].diff()
        df["Northing_Diff"] = df["Northing"].diff()
        df["Distance"] = np.sqrt(df["Easting_Diff"] ** 2 + df["Northing_Diff"] ** 2)
        df["Velocity"] = df["Distance"] / df["Time_Diff"] * 1.94384  # Constant to convert m/s to knots
        df["Avg_Velocity"] = df["Velocity"].rolling(window=self.avg_vel_segment_len).mean()

        self.avg_vel = df["Velocity"].mean()
        self.vel_cutoff = self.avg_vel + self.vel_cutoff_offset

        return df

    # Prefilter data based on the velocity threshold and return both filtered and outlier data
    def prefilter_data(self, df):
        df = self.df_append_calc_vel(df)

        prefilter_vel_high_cutoff = self.avg_vel * self.prefilter_avg_vel_mult
        prefilter_vel_low_cutoff = self.avg_vel / self.prefilter_avg_vel_mult

        outlier_mask = (df["Velocity"] > prefilter_vel_high_cutoff) & (df["Velocity"].shift(-1) > prefilter_vel_high_cutoff) | (df["Velocity"] < prefilter_vel_low_cutoff) & (
            df["Velocity"].shift(-1) < prefilter_vel_low_cutoff
        )

        df_prefiltered = df[~outlier_mask]
        df_outlier_prefilter = df[outlier_mask]

        return df_prefiltered, df_outlier_prefilter

    # Filter data based on the velocity threshold and combine outliers
    def filter_data(self, df):
        # ----------- Prefilter -----------
        self.df_prefiltered, self.outliers_dict["prefilter"] = self.prefilter_data(df.copy())
        self.df_prefiltered = self.df_append_calc_vel(self.df_prefiltered)

        # ----------- Filter -----------
        outlier_mask = (self.df_prefiltered["Velocity"] > self.vel_cutoff) & (self.df_prefiltered["Velocity"].shift(-1) > self.vel_cutoff)
        self.df_filtered = self.df_prefiltered[~outlier_mask]
        self.outliers_dict["filter"] = self.df_prefiltered[outlier_mask]

        return self.df_filtered, self.outliers_dict

    # Calculate statistics like total distance, average velocity, max velocity, etc., and add number of points
    def calculate_statistics(self, df):
        df = self.df_append_calc_vel(df.copy())

        valid_velocities = df["Velocity"].dropna()
        valid_distances = df["Distance"].dropna()
        total_distance = valid_distances.sum()
        avg_vel = valid_velocities.mean()
        max_vel = valid_velocities.max()
        travel_heading = np.degrees(np.arctan2(df["Northing_Diff"].sum(), df["Easting_Diff"].sum()))
        total_time = df["Time_Diff"].sum()
        dist_over_time_knots = total_distance / total_time * 1.94384  # In knots
        num_points = len(df)

        stats = {
            "Total Distance (m)": total_distance,
            "Average Velocity (knots)": avg_vel,
            "Maximum Velocity (knots)": max_vel,
            "Travel Heading (degrees)": travel_heading,
            "Total Time (s)": total_time,
            "Distance over Time (knots)": dist_over_time_knots,
            "Number of Points": num_points,
        }
        return stats

    def save_data(self, df, folder, filename, columns=None):
        if not os.path.exists(folder):
            os.makedirs(folder)

        output_path = os.path.join(folder, filename)
        if columns is not None:
            df[columns].to_csv(output_path, index=False, float_format="%.4f")
        else:
            df.to_csv(output_path, index=False, float_format="%.4f")

    # Plot data: trajectory and velocity graph including cutoff line and number of filtered points
    def plot_data_and_vel(self, df, filename, label, ax1, ax2):
        ax1.clear()
        ax2.clear()

        # ------------ PLOT #1 Data with all Outliers ------------
        ax1.plot(df["Easting"], df["Northing"], marker="o", linestyle="-", label=label)

        sizes = [30, 20, 10]
        colors_rgba = [(1, 0, 0, 1), (1, 0.5, 0, 1), (1, 1, 0, 1), (0, 0, 0, 1)]  # RGBA
        for idx, (outlier_type, outlier_df) in enumerate(self.outliers_dict.items()):
            if not outlier_df.empty:
                ax1.scatter(
                    outlier_df["Easting"], outlier_df["Northing"], color=colors_rgba[idx], s=sizes[idx], label=f"{outlier_type} Outliers ({len(outlier_df)} points)", zorder=10
                )
        ax1.set_title(f"Data - {filename}")
        ax1.set_xlabel("Easting")
        ax1.set_ylabel("Northing")
        ax1.legend()

        # ------------ PLOT #2 Data Velocity with all Outliers ------------
        ax2.plot(df["Easting"], df["Velocity"], "b-", label="Velocity")

        for idx, (outlier_type, outlier_df) in enumerate(self.outliers_dict.items()):
            if not outlier_df.empty:
                ax2.scatter(
                    outlier_df["Easting"], outlier_df["Velocity"], color=colors_rgba[idx], s=sizes[idx], label=f"{outlier_type} Outliers ({len(outlier_df)} points)", zorder=10
                )

        ax2.axhline(y=self.vel_cutoff, color="r", linestyle="--", label=f"Cutoff: {self.vel_cutoff:.2f} knots")
        ax2.axhline(y=self.avg_vel, color="k", linestyle=":", label=f"Avg Velocity: {self.avg_vel:.2f} knots", alpha=0.7)
        ax2.set_title(f"Velocity - {filename}")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Velocity (knots)")
        ax2.legend()
        
        # ------------ Auto Zoom (Works on slider move and matplotlib window zoom) ------------

        def on_xlims_change(event_ax):
            x_min, x_max = event_ax.get_xlim()
            if event_ax == ax1:
                northing_min = np.nanmin(df["Northing"][df["Easting"].between(x_min, x_max)])
                northing_max = np.nanmax(df["Northing"][df["Easting"].between(x_min, x_max)])
                ax1.set_ylim(northing_min, northing_max)
            elif event_ax == ax2:
                velocity_min = np.nanmin(df["Velocity"][df["Easting"].between(x_min, x_max)])
                velocity_max = np.nanmax(df["Velocity"][df["Easting"].between(x_min, x_max)])
                ax2.set_ylim(velocity_min, velocity_max)

        ax1.callbacks.connect('xlim_changed', on_xlims_change)
        ax2.callbacks.connect('xlim_changed', on_xlims_change)

        # Call on_xlims_change initially to set the y-limits based on the initial x-limits
        on_xlims_change(ax1)
        on_xlims_change(ax2)

    def process_data(self, df):
        # ------------- Filtering -------------
        self.df_original = df.copy()
        self.df_filtered, self.outliers_dict = self.filter_data(self.df_original.copy())

        # ----------- Compare stats -----------
        stats_list = pd.DataFrame({"Original": self.calculate_statistics(self.df_original), "Prefiltered": self.calculate_statistics(self.df_prefiltered)})
        print(stats_list)
        
    def visualize_data(self, filename):
        # ------------- Visualization Plotting -------------
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8), sharex=True)

        def plot_all():
            nonlocal ax1, ax2, ax3, ax4
            self.plot_data_and_vel(self.df_append_calc_vel(self.df_original), filename, "Original Data", ax1, ax2)
            self.plot_data_and_vel(self.df_filtered, filename, "Filtered Data", ax3, ax4)
            
            if self.zoom_ax_x != (None, None):
                ax1.set_xlim(self.zoom_ax_x)
            
        plot_all()

        # ------------- Slider -------------

        ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03])
        slider = Slider(ax_slider, "Velocity Threshold", -10.0, 10.0, valinit=self.vel_cutoff_offset)

        # Store zoom levels and update plot
        def on_slider_update(val):
            self.vel_cutoff_offset = slider.val
            self.vel_cutoff = self.avg_vel + self.vel_cutoff_offset
            df_filtered, _ = self.filter_data(df)

            # Store zoom positions
            self.zoom_ax_x = ax1.get_xlim()
            self.zoom_ax1_y = ax1.get_ylim()
            self.zoom_ax2_y = ax2.get_ylim()
            self.zoom_ax3_y = ax3.get_ylim()
            self.zoom_ax4_y = ax4.get_ylim()

            plot_all()
            
            fig.canvas.draw_idle()

        slider.on_changed(on_slider_update)

        plt.show()

    def calculate_best_fit(self, df):
        total_points = len(df)
        cut_off = int(total_points * 0.1)  # 10% cutoff
        df_trimmed = df.iloc[cut_off:total_points - cut_off]

        easting = df_trimmed['Easting'].values.reshape(-1, 1)  # Reshaped for sklearn
        northing = df_trimmed['Northing'].values

        reg = LinearRegression().fit(easting, northing)
        slope = reg.coef_[0]
        intercept = reg.intercept_

        plt.scatter(easting, northing, label="Trimmed Data Points", color="blue")
        
        easting_range = np.linspace(easting.min(), easting.max(), 100)
        northing_fit = slope * easting_range + intercept
        plt.plot(easting_range, northing_fit, label=f"Best Fit: y = {slope:.4f}x + {intercept:.4f}", color="red")

        plt.xlabel('Easting')
        plt.ylabel('Northing')
        plt.title('Best Fit Line for Northing vs Easting')
        plt.legend()
        plt.show()

        return slope, intercept


    # Main processing loop for all files
    def process_files(self):
        self.load_gps_data()

        for filename, df, columns in self.gps_data_list:
            print(f"\nWorking on a file:   {filename}")

            self.process_data(df)
            self.visualize_data(filename)
            
            
            slope, intercept = self.calculate_best_fit(self.df_filtered)
            print(f"Slope: {slope}")
            print(f"Intercept: {intercept}")
            
            

            # TODO
            # self.save_data(df, self.path_out_debug, filename)
            # self.save_data(df, self.path_out_filtered, filename, columns)

    

# Run the processing
if __name__ == "__main__":
    processor = GPSDataProcessor(data_folder="./data", output_folder="./data_filtered", avg_vel_segment_len=20, vel_cutoff_offset=1)
    processor.process_files()
