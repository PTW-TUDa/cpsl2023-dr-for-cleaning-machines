from __future__ import annotations

import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import timedelta
import matplotlib as mpl
from matplotlib import cm
import matplotlib.dates as mdates

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# base_path is a string that contains the folder of the files to be evaluated.
base_path = "D:/Documents"

# files contains the names of the files to be evaluated. Format:
# files = ["KEA_221118_controlExperiment1.csv", "exp_221118_000_01.csv", "exp_221118_000_01_modellog.hdf"] 
files = ["KEA_221118_controlExperiment1.csv", "exp_221118_000_01.csv", "exp_221118_000_01_modellog.hdf"]

measure_interval = 1
model_sampling_time = 10
margins = 100

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = "9"
linestyles = [":","--","-"]
greys = lambda x: cm.Greys(int(255-((255-100)/5)*x))

plt.rcParams['text.usetex'] = True # interpret TeX commands

def main():
    df_measurement, df_model_meas, df_model_pred = read_data(*[os.path.join(base_path, path) for path in files])

    measure_starting_point = find_starting_point(df_measurement.bAlgorithmModeActivated)
    all_measures = df_measurement.join(df_convert_to_timeindex_and_resample(
        df_model_meas,
        df_measurement.index[measure_starting_point],
        model_sampling_time,
        measure_interval,
        ["durationDrying", "temp_tank", "t_environment", "durationCleaning", "durationStart"]
    ), how="left", rsuffix="_model").dropna()

    total_steps = len(df_model_pred.index.get_level_values("step").drop_duplicates())
    prediction_scope = len(df_model_pred.index.get_level_values("index").drop_duplicates())
    df_model_pred.set_index(
        pd.MultiIndex.from_product(
            (range(total_steps), measured_date_range(all_measures.index[0], model_sampling_time, prediction_scope)),
            names=("step", "index")
        ),
        inplace=True
    )
    df_model_pred = df_model_pred.reindex(
        pd.MultiIndex.from_product((range(total_steps), all_measures.index), names=("step", "index")),
        fill_value=float("nan")
    ).ffill()

    # Prepare values for the operating state of the cleaning machine
    all_measures.loc[
        ((all_measures.nKEAOperatingState == 1) | (all_measures.nKEAOperatingState == 2)),
        "nKEAOperatingState"
    ] = 0
    all_measures.loc[all_measures.nKEAOperatingState == 9, "nKEAOperatingState"] = 1
    all_measures.loc[
        ((all_measures.nKEAOperatingState == 3)
        | (all_measures.nKEAOperatingState == 4)
        | (all_measures.nKEAOperatingState == 5)
        | (all_measures.nKEAOperatingState == 6)),
        "nKEAOperatingState"
    ] = 2
    all_measures.loc[
        ((all_measures.nKEAOperatingState == 7) | (all_measures.nKEAOperatingState == 8)),
        "nKEAOperatingState"
    ] = 3

    all_measures.loc[(all_measures.bSetStatusOnAlgorithm == True), "bSetStatusOnAlgorithm"] = 1
    all_measures.loc[(all_measures.bSetStatusOnAlgorithm == False), "bSetStatusOnAlgorithm"] = 0

    figure_subplots = [3]
    figures = [plt.figure(i, figsize=(7.5, 5), dpi=250, layout="tight") for i in range(len(figure_subplots))]
    axes = [figures[i].subplots(num) for i, num in enumerate(figure_subplots)]

    x = all_measures.index
    starttime, endtime = x[1] - timedelta(seconds=margins), x[-1] + timedelta(seconds=margins)

    # 1) Plot predicted and measured operating state of the machine
    lines = []
    axes2twin = axes[0][0].twinx()
    lines.append(axes[0][0].plot(x, all_measures.fRealPower / 1000, color=greys(0), linestyle="--", label="power consumption")[0])
    lines.append(
        plot_predictions(axes2twin, x, df_model_pred["c"]*100000, color=greys(4), linestyle="-", prediction_scope=50, label="energy price $c_k$")[0]
    )
    format_axes(
        axes[0][0], grid=True, xticklabels=[], xlim=(starttime, endtime),
        ylabel="real power in kW"
    )
    format_axes(axes2twin, ylabel="energy price in €/MWh", legend="upper right", legend_items=lines, legend_bbox_to_anchor=(0.985, 0.95))

    # 2) Plot predicted and measured operating state of the machine
    lines = []
    lines.append(axes[0][1].plot(x, all_measures.nKEAOperatingState, color=greys(0), linestyle="-", label="machine state $n_{\r start}$")[0])
    lines.append(
        plot_predictions(axes[0][1], x, df_model_pred["i"], color=greys(3), linestyle=":", prediction_scope=50, label="interruption state")[0]
    )
    format_axes(
        axes[0][1], grid=True, xlim=(starttime, endtime), xticklabels=[],
        ylabel="machine state", ylim=(-0.2, 3.2), yticks=[0, 1, 2, 3],
        yticklabels=["operational", "interrupted", "cleaning", "drying"], legend="upper left", legend_items=lines,
        legend_bbox_to_anchor = (0.002, 0.9999)
    )

    # 3) Plot predicted and measured temperature in the machine's tank
    lines = []
    axes0twin = axes[0][2].twinx()
    lines.append(
        plot_predictions(axes[0][2], x, df_model_pred["t"], color=greys(5), linestyle=":", prediction_scope=200, label="temperature prediction $t_0$ to $t_{200}$")[0]
    )
    lines.append(
        axes[0][2].plot(x, all_measures.fTankTemperature, color=greys(0), linestyle="-", label="tank temperature $t_{\r start}$")[0]
    )
    lines.append(
        axes0twin.plot(x, all_measures.bSetStatusOnAlgorithm, color=greys(2), linestyle="--", label="tank heater state")[0]
    )

    format_axes(
        axes[0][2], grid=True, ylabel="tank temp. in °C", ylim=(55, 62), xlim=(starttime, endtime), xlabel="time",
        legend="upper right", legend_items=lines, legend_bbox_to_anchor=(0.998, 0.57)
    )
    axes[0][2].xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    axes[0][2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    format_axes(axes0twin, ylabel="tank heater state", ylim=(-0.2, 2.6), yticks=[0, 1], yticklabels=["off", "on"])
    

    plt.savefig("auswertung.svg")
    plt.savefig("auswertung.png")
    plt.show()


def read_data(measure_file: str, model_state_file: str, predictions_file: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(measure_file, index_col=0, parse_dates=True),
        pd.read_csv(model_state_file, delimiter=";"),
        pd.read_hdf(predictions_file)
    )


def plot_predictions(ax: Axes, x, y, *, prediction_scope: int | None = None, mark_first = False, **kwargs):
    for idx in range(int(len(x)/10)):
        hdl = ax.plot(
            (x + timedelta(seconds=idx*10))[0:prediction_scope],
            y.loc[idx].values[0:prediction_scope],
            **kwargs
        )

        if mark_first:
            hdl = ax.plot((x + timedelta(seconds=idx*10))[1], y.loc[idx].values[1], 'k*')

    return hdl


def find_starting_point(algorithm_mode_series: pd.Series):
    for idx, val in enumerate(algorithm_mode_series.values):
        if not algorithm_mode_series.iloc[idx] and algorithm_mode_series.iloc[idx+1]:
            return idx+1


def df_convert_to_timeindex_and_resample(df, time_start, sampling_time, resampling_time, interpolate_cols = None):
    _sampling_time = sampling_time if isinstance(sampling_time, timedelta) else timedelta(seconds=sampling_time)
    _resampling_time = resampling_time if isinstance(resampling_time, timedelta) else timedelta(seconds=resampling_time)

    df.set_index(measured_date_range(time_start, _sampling_time, len(df)), inplace=True)
    df = df.resample(_resampling_time).asfreq()

    if interpolate_cols is not None:
        df[interpolate_cols] = df[interpolate_cols].interpolate(axis=0)
    df.ffill(axis=0, inplace=True)
    return df


def measured_date_range(time_start, sampling_time, length):
    _sampling_time = sampling_time if isinstance(sampling_time, timedelta) else timedelta(seconds=sampling_time)
    return pd.date_range(
        time_start,
        time_start + timedelta(seconds=length * _sampling_time.total_seconds()) - _sampling_time,
        freq=_sampling_time
    )


def format_axes(
    ax, *, xlabel = None, xlim = None, xticks = None, xticklabels=None,
    ylabel = None, ylim = None, yticks = None, yticklabels=None, grid = False,
    legend = None, legend_items=None, legend_bbox_to_anchor=None
):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if grid:
        ax.grid(grid, color='gray', linestyle='dashed')
    else:
        ax.grid(False)

    if legend is not None:
        legend_kwargs = {"loc": legend}
        if legend_items is not None:
            legend_kwargs["handles"] = legend_items
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        ax.legend(borderaxespad=0.25, **legend_kwargs)


if __name__ == "__main__":
    main()
