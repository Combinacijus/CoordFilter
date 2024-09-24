import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob
import os
from segment_filter import SegmentFilter
from binary_search import binary_search


class GPSDataProcessor:
    def __init__(self, data_folder, output_folder, avg_vel_segment_len):
        # Filter parameters
        self.prefilter_avg_vel_mult = 5.0  # If vel is bigger or smaller by this multiplier then delete in prefilter
        self.default_vel_filter_cutoff_percentage = 1.0  # Filter out this percentage of velocity outliers
        self.vel_filter_step = 0.001  # While using search what's the smallest step in vel offset
        self.vel_filter_min_vel_offset = -50
        self.vel_filter_max_vel_offset = 50
        self.segment_filter_points_per_segment = 20

        # Parameters
        self.data_folder = data_folder
        self.path_out_filtered = output_folder
        self.path_out_debug = output_folder + "_debug"
        self.avg_vel_segment_len = avg_vel_segment_len
        self.vel_cutoff_offset = None
        self.vel_cutoff_percentage = self.default_vel_filter_cutoff_percentage
        self.avg_vel = 0
        self.vel_cutoff = 0
        self.gps_data_list = []
        self.zoom_ax_x = None

        self.df_original = None
        self.df_original_vel = None
        self.df_prefiltered = None
        self.df_filtered = None

        self.outliers_dict = {}  # Dictionary to hold different types of outliers

    def load_gps_data(self):
        expected_columns = ["Date", "Time", "Easting", "Northing"]

        # Get the file paths for all .txt files in the data folder
        file_paths = glob.glob(os.path.join(self.data_folder, "*.txt"))
        print(f"Files loaded:\n{file_paths}")

        for file_path in file_paths:
            # First try to load the file, assuming it has a header
            df = pd.read_csv(file_path, header=0)
            columns = df.columns.tolist()

            # Check if the first row is actual data (should be numeric for Easting/Northing)
            first_row_is_data = all(pd.api.types.is_numeric_dtype(df[col]) for col in columns if col in ["Easting", "Northing"])

            if columns != expected_columns and first_row_is_data:
                # If the header is actually the first line of data, reload assuming no header
                print(f"Warning: First row appears to be data in {os.path.basename(file_path)}. Assuming default columns.")
                df = pd.read_csv(file_path, header=None, names=expected_columns)
            elif columns != expected_columns:
                # If the columns don't match and it's not a missing header scenario, raise an error
                raise ValueError(f"Unexpected header in file {os.path.basename(file_path)}: {columns}. " f"Expected columns: {expected_columns}")

            df.dropna(inplace=True) #  Drop rows with any missing values in any column

            # Save the columns and append the data to gps_data_list
            self.gps_data_list.append((os.path.basename(file_path), df, expected_columns))

            # Combine 'Date' and 'Time' columns into a datetime column
            df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

    def df_append_calc_vel(self, df):
        df.loc[:, "Time_Diff"] = df["Datetime"].diff().dt.total_seconds()
        df.loc[:, "Easting_Diff"] = df["Easting"].diff()
        df.loc[:, "Northing_Diff"] = df["Northing"].diff()
        df.loc[:, "Distance"] = np.sqrt(df["Easting_Diff"] ** 2 + df["Northing_Diff"] ** 2)
        df.loc[:, "Velocity"] = df["Distance"] / df["Time_Diff"] * 1.94384  # Convert m/s to knots
        # df.loc[:, "Avg_Velocity"] = df["Velocity"].rolling(window=self.avg_vel_segment_len).mean()

        self.avg_vel = df["Velocity"].mean()

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

    def apply_velocity_filter(self, df, *, cutoff_percentage=None, cutoff_vel_offset=None):
        self.vel_cutoff = self.update_vel_cutoff(cutoff_percentage=cutoff_percentage, cutoff_offset=cutoff_vel_offset)

        outlier_mask = (df["Velocity"] > self.vel_cutoff) & (df["Velocity"].shift(-1) > self.vel_cutoff)
        self.df_filtered = df[~outlier_mask]
        self.outliers_dict["filter"] = df[outlier_mask]

        return self.df_filtered, self.outliers_dict

    def get_vel_outlier_percentage(self, offset):
        df_filtered, outliers_dict = self.apply_velocity_filter(self.df_prefiltered.copy(), cutoff_vel_offset=offset)

        outlier_count = len(outliers_dict["filter"])
        total_count = len(df_filtered) + outlier_count
        outlier_percentage = (outlier_count / total_count) * 100 if total_count > 0 else 0

        if total_count < 0:
            raise Exception("Error: No Data")

        return outlier_percentage

    def binary_search_vel_filter_offset(self, target_percentage):
        vel_offset = binary_search(self.vel_filter_min_vel_offset, self.vel_filter_max_vel_offset, self.get_vel_outlier_percentage, target_percentage, self.vel_filter_step, False)
        return vel_offset

    def calc_and_plot_outliers_vs_offset_graph(self, step=0.01):
        outlier_percentages = []
        derivative_outlier_percentages = []

        def get_auto_range(target_percentage):
            target_percentage1 = target_percentage
            target_percentage2 = 100 - target_percentage

            # Automatically find min and max offset to plot
            offset_low = self.binary_search_vel_filter_offset(target_percentage1)
            offset_mid = self.binary_search_vel_filter_offset(50)
            offset_high = self.binary_search_vel_filter_offset(target_percentage2)

            distance_to_min = abs(offset_mid - offset_low)
            distance_to_max = abs(offset_mid - offset_high)
            offset_low = offset_mid - min(distance_to_min, distance_to_max)
            offset_high = offset_mid + min(distance_to_min, distance_to_max)

            if self.get_vel_outlier_percentage(offset_low) > self.get_vel_outlier_percentage(offset_high):
                offset_low, offset_high = offset_high, offset_low

            print()
            print(f"f({offset_low}) = {self.get_vel_outlier_percentage(offset_low)} -> ({target_percentage1} targeted)")
            print(f"f({offset_mid}) = {self.get_vel_outlier_percentage(offset_mid)} -> ({50} targeted)")
            print(f"f({offset_high}) = {self.get_vel_outlier_percentage(offset_high)} -> ({target_percentage2} targeted)")

            offset_min = min(offset_low, offset_high)
            offset_max = max(offset_low, offset_high)

            return np.arange(offset_min, offset_max + step, step)

        offset_list = get_auto_range(target_percentage=0.5)

        # Calculate velocity outlier percentages
        for offset in offset_list:
            df_filtered, outliers_dict = self.apply_velocity_filter(self.df_prefiltered.copy(), cutoff_vel_offset=offset)
            points_count = len(df_filtered)
            outlier_count = len(outliers_dict["filter"])
            total_count = points_count + outlier_count
            outlier_percentage = (outlier_count / total_count) * 100 if total_count > 0 else 0
            outlier_percentages.append(outlier_percentage)

        # Calculate derivative of outlier percentages
        for i in range(1, len(outlier_percentages)):
            derivative = (outlier_percentages[i] - outlier_percentages[i - 1]) / (offset_list[i] - offset_list[i - 1])
            derivative_outlier_percentages.append(derivative)

        # Plotting offset vs outlier percentage and its derivative
        fig, axs = plt.subplots(2, figsize=(10, 8), sharex=True)

        # Plot outlier percentages
        axs[0].plot(offset_list, outlier_percentages, marker="o", color="b", label="Outlier Percentage")
        axs[0].set_xlabel("Offset")
        axs[0].set_ylabel("Outlier Percentage (%)")
        axs[0].set_title("Offset vs Outlier Percentage")
        axs[0].grid(True)
        axs[0].legend()

        # Plot derivative of outlier percentages
        axs[1].plot(offset_list[1:], derivative_outlier_percentages, marker="o", color="g", label="Derivative of Outlier Percentage")
        axs[1].set_xlabel("Offset")
        axs[1].set_ylabel("Derivative of Outlier Percentage")
        axs[1].set_title("Offset vs Derivative of Outlier Percentage")
        axs[1].grid(True)
        axs[1].legend()

        # Add slider for target percentage
        ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03])
        slider = Slider(ax_slider, "Velocity Filter Percentage", 0.0, 5.0, valinit=self.vel_cutoff_percentage)

        # Initialize horizontal and vertical lines as None
        line_h1 = None
        line_v1 = None
        line_h2 = None
        line_v2 = None

        def update_plot(target_percentage):
            nonlocal line_h1, line_v1, line_h2, line_v2

            line_h_posx = offset_list[np.abs(np.array(outlier_percentages) - target_percentage).argmin()]
            line_v2_posy = np.interp(line_h_posx, offset_list[1:], derivative_outlier_percentages)

            # If the lines exist, remove them first
            if line_h1 is not None:
                line_h1.remove()
            if line_v1 is not None:
                line_v1.remove()
            if line_h2 is not None:
                line_h2.remove()
            if line_v2 is not None:
                line_v2.remove()

            # Redraw horizontal and vertical lines at new slider values
            line_h1 = axs[0].axhline(y=target_percentage, color="r", linestyle="--")
            line_v1 = axs[0].axvline(x=line_h_posx, color="r", linestyle="--")
            line_h2 = axs[1].axhline(y=line_v2_posy, color="r", linestyle="--")
            line_v2 = axs[1].axvline(x=line_h_posx, color="r", linestyle="--")

            fig.canvas.draw_idle()

        def on_slider_update(slider_val):
            self.vel_cutoff = self.update_vel_cutoff(cutoff_percentage=slider_val)
            update_plot(slider_val)

        slider.on_changed(on_slider_update)
        update_plot(slider.val)

        fig.suptitle("Speed offset and Filtered Out Outlier Percentage Analysis")
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
        # ax1.lines.remove()
        # ax2.lines.remove()

        # ------------ PLOT #1 Data with all Outliers ------------
        ax1.plot(df["Easting"], df["Northing"], marker="o", linestyle="-", label=f"{label} (Total Points: {len(df)})")

        sizes = [8, 7, 6]
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
        ax2.axhline(y=self.avg_vel, color="k", linestyle=":", label=f"Avg Velocity: {self.avg_vel:.2f} knots")
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
        # ------------- Initial Filtering -------------
        self.df_original = df.copy()
        self.df_original_vel = self.df_append_calc_vel(self.df_original.copy())
        self.df_prefiltered, self.outliers_dict["prefilter"] = self.prefilter_data(self.df_original_vel)
        self.df_filtered, self.outliers_dict = self.apply_velocity_filter(self.df_prefiltered.copy(), cutoff_percentage=self.default_vel_filter_cutoff_percentage)
        # ----------- Compare stats -----------
        # stats_list = pd.DataFrame(
        #     {
        #         "Original": self.calculate_statistics(self.df_original),
        #         "Prefiltered": self.calculate_statistics(self.df_prefiltered),
        #         "Filtered": self.calculate_statistics(self.df_filtered),
        #     }
        # )
        # print(stats_list)

    def update_vel_cutoff(self, *, cutoff_percentage=None, cutoff_offset=None):
        if cutoff_percentage is not None and cutoff_offset is not None:
            raise ValueError("Only one of cutoff_percentage or cutoff_vel_offset should be defined.")

        if cutoff_percentage is not None:
            self.vel_cutoff_percentage = cutoff_percentage
            self.vel_cutoff_offset = self.binary_search_vel_filter_offset(cutoff_percentage)
        elif cutoff_offset is not None:
            self.vel_cutoff_offset = cutoff_offset

        self.vel_cutoff = self.avg_vel + self.vel_cutoff_offset

        return self.vel_cutoff

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

        # ------------- Slider Visualize -------------
        def on_slider_update(val):
            self.df_filtered, self.outliers_dict = self.apply_velocity_filter(self.df_prefiltered.copy(), cutoff_percentage=val)
            plot_all()

        ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03])
        slider = Slider(ax_slider, "Velocity Filter Percentage", 0.0, 5.0, valinit=self.vel_cutoff_percentage)
        slider.on_changed(on_slider_update)

        plt.show()

    # Main processing loop for all files
    def process_files(self):
        self.load_gps_data()

        for filename, df, columns in self.gps_data_list:
            print(f"\nWorking on a file:   {filename}")

            self.process_data(df)
            self.calc_and_plot_outliers_vs_offset_graph()
            self.visualize_data(filename)

            import segment_filter
            segment_filter = SegmentFilter(df, self.segment_filter_points_per_segment)
            segment_filter.calculate_best_fit()
            
            # TODO Both won't work at the same time
            # segment_filter.plot_offset_vs_outlier_percentage(min_offset=0, max_offset=2, step=0.1)
            segment_filter.plot()
            
            # TODO 
            # 1. plot_offset_vs_outlier_percentage() and segment_filter.plot() does not work if both uncommented
            # 2. Make segment_filter to choose by what percentage to filter out and by min distance whichever keeps more
            #    Use as example binary_search_vel_filter_offset() and get_auto_range()
            # 3. Put every filter in pipeline to automatically filter and save everything (make option to skip GUI)

            # TODO
            # self.save_data(df, self.path_out_debug, filename)
            # self.save_data(df, self.path_out_filtered, filename, columns)


if __name__ == "__main__":
    processor = GPSDataProcessor(data_folder="./data", output_folder="./data_filtered", avg_vel_segment_len=20)

    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()

    processor.process_files()

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(20)
