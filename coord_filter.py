import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob
import os
from segment_filter import SegmentFilter


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
        self.zoom_ax_x = None

        self.prefilter_avg_vel_mult = 5.0  # If vel is bigger or smaller by this value then delete in prefilter

        self.df_original = None
        self.df_original_vel = None
        self.df_prefiltered = None
        self.df_filtered = None

        self.outliers_dict = {}  # Dictionary to hold different types of outliers

    def load_gps_data(self):
        file_paths = glob.glob(os.path.join(self.data_folder, "*.txt"))
        print(f"Files loaded:\n{file_paths}")

        for file_path in file_paths:
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()  # Save list of columns
            self.gps_data_list.append((os.path.basename(file_path), df, columns))

            df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

    def df_append_calc_vel(self, df):
        df.loc[:, "Time_Diff"] = df["Datetime"].diff().dt.total_seconds()
        df.loc[:, "Easting_Diff"] = df["Easting"].diff()
        df.loc[:, "Northing_Diff"] = df["Northing"].diff()
        df.loc[:, "Distance"] = np.sqrt(df["Easting_Diff"] ** 2 + df["Northing_Diff"] ** 2)
        df.loc[:, "Velocity"] = df["Distance"] / df["Time_Diff"] * 1.94384  # Convert m/s to knots
        # df.loc[:, "Avg_Velocity"] = df["Velocity"].rolling(window=self.avg_vel_segment_len).mean()

        self.avg_vel = df["Velocity"].mean()
        self.vel_cutoff = self.avg_vel + self.vel_cutoff_offset

        return df

    def prefilter_data(self, df):
        if "Velocity" not in df.columns:
            df = self.df_append_calc_vel(df)

        prefilter_vel_high_cutoff = self.avg_vel * self.prefilter_avg_vel_mult
        prefilter_vel_low_cutoff = self.avg_vel / self.prefilter_avg_vel_mult

        outlier_mask = ((df["Velocity"] > prefilter_vel_high_cutoff) & (df["Velocity"].shift(-1) > prefilter_vel_high_cutoff)) | (
            (df["Velocity"] < prefilter_vel_low_cutoff) & (df["Velocity"].shift(-1) < prefilter_vel_low_cutoff)
        )

        df_prefiltered = df[~outlier_mask]
        df_outlier_prefilter = df[outlier_mask]

        df_prefiltered = self.df_append_calc_vel(df_prefiltered)  # Recalculate velocities

        return df_prefiltered, df_outlier_prefilter

    def filter_data(self, df):
        outlier_mask = (self.df_prefiltered["Velocity"] > self.vel_cutoff) & (self.df_prefiltered["Velocity"].shift(-1) > self.vel_cutoff)
        self.df_filtered = self.df_prefiltered[~outlier_mask]
        self.outliers_dict["filter"] = self.df_prefiltered[outlier_mask]

        return self.df_filtered, self.outliers_dict

    def calc_and_plot_outliers_vs_offset_graph(self, min_offset=-2.5, max_offset=0, step=0.02):
        offset_list = np.arange(min_offset, max_offset + step, step)
        outlier_percentages = []
        derivative_outlier_percentages = []
        for offset in offset_list:
            self.vel_cutoff = self.avg_vel + offset
            df_filtered, outliers_dict = self.filter_data(self.df_prefiltered.copy())
            points_count = len(df_filtered)
            outlier_count = len(self.outliers_dict["filter"])
            total_count = points_count + outlier_count
            outlier_percentage = (outlier_count / total_count) * 100 if total_count > 0 else 0
            outlier_percentages.append(outlier_percentage)

        # Calculate derivative of outlier percentages
        for i in range(1, len(outlier_percentages)):
            derivative = (outlier_percentages[i] - outlier_percentages[i - 1]) / (offset_list[i] - offset_list[i - 1])
            derivative_outlier_percentages.append(derivative)

        # Plotting offset vs outlier percentage and its derivative on different subplots
        fig, axs = plt.subplots(2, figsize=(10, 8), sharex=True)
        axs[0].plot(offset_list, outlier_percentages, marker="o", color="b", label="Outlier Percentage")
        axs[0].set_xlabel("Offset")
        axs[0].set_ylabel("Outlier Percentage (%)")
        axs[0].set_title("Offset vs Outlier Percentage")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(offset_list[1:], derivative_outlier_percentages, marker="o", color="r", label="Derivative of Outlier Percentage")
        axs[1].set_xlabel("Offset")
        axs[1].set_ylabel("Derivative of Outlier Percentage")
        axs[1].set_title("Offset vs Derivative of Outlier Percentage")
        axs[1].grid(True)
        axs[1].legend()

        # Calculate top 20%, 10%, 5% values and mean
        top_20_percent_value = np.percentile(outlier_percentages, 80)
        top_10_percent_value = np.percentile(outlier_percentages, 90)
        top_5_percent_value = np.percentile(outlier_percentages, 95)
        mean_value = np.mean(outlier_percentages)

        print(f"Top 20% Value: {top_20_percent_value}")
        print(f"Top 10% Value: {top_10_percent_value}")
        print(f"Top 5% Value: {top_5_percent_value}")
        print(f"Mean Value: {mean_value}")

        plt.tight_layout()
        plt.show()

    def calculate_statistics(self, df):
        if "Velocity" not in df.columns:
            df = self.df_append_calc_vel(df.copy())

        valid_velocities = df["Velocity"].dropna()
        valid_distances = df["Distance"].dropna()
        total_distance = valid_distances.sum()
        avg_vel = valid_velocities.mean()
        max_vel = valid_velocities.max()
        travel_heading = np.degrees(np.arctan2(df["Northing_Diff"].sum(), df["Easting_Diff"].sum()))
        self.travel_heading = travel_heading
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
        ax1.plot(df["Easting"], df["Northing"], marker="o", linestyle="-", label=f"{label} (Total Points: {len(df)})")

        sizes = [30, 20, 10]
        colors_rgba = [(1, 0, 0, 1), (1, 0.5, 0, 1), (1, 1, 0, 1), (0, 0, 0, 1)]  # RGBA
        for idx, (outlier_type, outlier_df) in enumerate(self.outliers_dict.items()):
            if not outlier_df.empty:
                ax1.scatter(
                    outlier_df["Easting"],
                    outlier_df["Northing"],
                    marker="x",
                    color=colors_rgba[idx],
                    s=sizes[idx],
                    label=f"{outlier_type} Outliers ({len(outlier_df)} points)",
                    zorder=10,
                )
        ax1.set_title(f"Data - {filename}")
        ax1.set_xlabel("Easting")
        ax1.set_ylabel("Northing")
        ax1.legend()

        # ------------ PLOT #2 Data Velocity with all Outliers ------------
        ax2.plot(df["Easting"], df["Velocity"], "b-", label=f"Velocity (Total Points: {len(df)})")

        for idx, (outlier_type, outlier_df) in enumerate(self.outliers_dict.items()):
            if not outlier_df.empty:
                ax2.scatter(
                    outlier_df["Easting"],
                    outlier_df["Velocity"],
                    marker="x",
                    color=colors_rgba[idx],
                    s=sizes[idx],
                    label=f"{outlier_type} Outliers ({len(outlier_df)} points)",
                    zorder=10,
                )

        ax2.axhline(y=self.vel_cutoff, color="r", linestyle="--", label=f"Cutoff: {self.vel_cutoff:.2f} knots")
        ax2.axhline(y=self.avg_vel, color="k", linestyle=":", label=f"Avg Velocity: {self.avg_vel:.2f} knots", alpha=0.7)
        ax2.set_title(f"Velocity - {filename}")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Velocity (knots)")
        ax2.legend()

    def ax_auto_y_zoom(self, df_list, ax_data_list, ax_vel_list):
        def on_xlims_change(event_ax):
            x_min, x_max = event_ax.get_xlim()

            for idx, ax in enumerate(ax_data_list):
                if ax == event_ax:
                    try:
                        northing_min = np.nanmin(df_list[idx]["Northing"][df_list[idx]["Easting"].between(x_min, x_max)])
                        northing_max = np.nanmax(df_list[idx]["Northing"][df_list[idx]["Easting"].between(x_min, x_max)])
                    except ValueError:
                        northing_min = np.nanmin(df_list[idx]["Northing"])
                        northing_max = np.nanmax(df_list[idx]["Northing"])
                    ax.set_ylim(northing_min, northing_max)

            for idx, ax in enumerate(ax_vel_list):
                if ax == event_ax:
                    try:
                        velocity_min = np.nanmin(df_list[idx]["Velocity"][df_list[idx]["Easting"].between(x_min, x_max)])
                        velocity_max = np.nanmax(df_list[idx]["Velocity"][df_list[idx]["Easting"].between(x_min, x_max)])
                    except ValueError:
                        velocity_min = 0
                        velocity_max = self.avg_vel * 1.5
                    ax.set_ylim(velocity_min, velocity_max)

        for ax in ax_data_list + ax_vel_list:
            ax.callbacks.connect("xlim_changed", on_xlims_change)

        for ax in ax_data_list + ax_vel_list:
            on_xlims_change(ax)

    def process_data(self, df):
        # ------------- Filtering -------------
        self.df_original = df.copy()
        self.df_original_vel = self.df_append_calc_vel(self.df_original.copy())
        self.df_prefiltered, self.outliers_dict["prefilter"] = self.prefilter_data(self.df_original_vel)
        self.df_filtered, self.outliers_dict = self.filter_data(self.df_prefiltered.copy())

        # ----------- Compare stats -----------
        stats_list = pd.DataFrame({"Original": self.calculate_statistics(self.df_original), "Prefiltered": self.calculate_statistics(self.df_prefiltered)})
        print(stats_list)

    def visualize_data(self, filename):
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey="row")
        self.ax_data_list = [self.ax1, self.ax2]
        self.ax_vel_list = [self.ax3, self.ax4]

        def plot_all():
            if not hasattr(self, "plotting_in_progress"):
                self.plotting_in_progress = False

            if self.plotting_in_progress is False:
                self.plotting_in_progress = True

                if len(self.ax1.lines) > 0:
                    self.zoom_ax_x = self.ax1.get_xlim()

                self.plot_data_and_vel(self.df_original_vel, filename, "Original Data", self.ax_data_list[0], self.ax_vel_list[0])
                self.plot_data_and_vel(self.df_filtered, filename, "Filtered Data", self.ax_data_list[1], self.ax_vel_list[1])

                # Restore zoom positions
                if self.zoom_ax_x is not None:
                    self.ax1.set_xlim(self.zoom_ax_x)

                self.ax_auto_y_zoom([self.df_original_vel, self.df_filtered], self.ax_data_list, [])
                # self.ax_auto_y_zoom([self.df_original_vel, self.df_filtered], [], self.ax_vel_list)

                self.fig.canvas.draw_idle()

                self.plotting_in_progress = False

        plot_all()
        # ------------- Slider -------------

        # Store zoom levels and update plot
        def on_slider_update(val):
            self.vel_cutoff_offset = slider.val
            self.vel_cutoff = self.avg_vel + self.vel_cutoff_offset
            self.df_filtered, self.outliers_dict = self.filter_data(self.df_prefiltered.copy())

            plot_all()

        ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03])
        slider = Slider(ax_slider, "Velocity Threshold", -10.0, 10.0, valinit=self.vel_cutoff_offset)
        slider.on_changed(on_slider_update)

        plt.show()

    # Main processing loop for all files
    def process_files(self):
        self.load_gps_data()

        for filename, df, columns in self.gps_data_list:
            print(f"\nWorking on a file:   {filename}")

            self.process_data(df)
            # self.calc_and_plot_outliers_vs_offset_graph()
            self.visualize_data(filename)

            # import segment_filter
            # points_per_segment=20
            # segment_filter = SegmentFilter(df, points_per_segment=points_per_segment)
            # segment_filter.calculate_best_fit()  # Calculate the best-fit lines
            # segment_filter.plot_offset_vs_outlier_percentage(min_offset=0, max_offset=2, step=0.1)
            # segment_filter.plot()

            # TODO
            # self.save_data(df, self.path_out_debug, filename)
            # self.save_data(df, self.path_out_filtered, filename, columns)


if __name__ == "__main__":
    processor = GPSDataProcessor(data_folder="./data", output_folder="./data_filtered", avg_vel_segment_len=20, vel_cutoff_offset=1)

    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()

    processor.process_files()

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(20)
