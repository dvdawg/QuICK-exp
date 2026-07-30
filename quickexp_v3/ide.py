"""Run-button helpers shared by the numbered experiment scripts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .backend import SyntheticBackend
from .config import ConfigRepository
from .data import ExperimentData
from .errors import ConfigError
from .flux_lookup import frequency_from_record
from .lab import connect_quick, make_held_flux_controller
from .runtime import CompletedRun, ExperimentRunner
from .util import to_builtin


def load_repository(project_root: Path) -> ConfigRepository:
    project_root = Path(project_root).resolve()

    def choose(stem: str) -> Path:
        configured = project_root / f"{stem}.yml"
        return (
            configured
            if configured.exists()
            else project_root / f"{stem}.example.yml"
        )

    repository = ConfigRepository.from_files(
        choose("hardware"),
        choose("calibration"),
        choose("presets"),
    )
    storage = repository.hardware.get("storage", {})
    storage_paths = {
        key: Path(storage[key])
        for key in ("root", "quick_native_root")
        if storage.get(key)
    }
    relative_paths = {
        key: value
        for key, value in storage_paths.items()
        if not value.is_absolute()
    }
    if relative_paths:
        hardware = deepcopy(repository.hardware)
        for key, value in relative_paths.items():
            hardware["storage"][key] = str((project_root / value).resolve())
        repository = ConfigRepository(
            hardware,
            repository.calibration,
            {"schema_version": 3, "presets": repository.presets},
            sources=repository.sources,
        )
    return repository


def print_offline_channel_expectation(repository: ConfigRepository) -> None:
    print("Configured channel expectation (live run verifies against soccfg):")
    for role in ("r", "rr", "q", "z"):
        channel = repository.hardware["channels"][role]
        endpoint = "ro" if role == "rr" else "gen"
        print(
            f"  {role:>2} = index {channel[endpoint]} -> "
            f"{channel.get('physical_port', 'unspecified')} "
            f"({channel.get('readout_type', channel.get('generator_type', 'any'))})"
        )


def inspect_connection(
    project_root: Path,
    *,
    live_hardware: bool,
    apply_rf_board: Optional[bool] = None,
) -> Any:
    repository = load_repository(project_root)
    if not live_hardware:
        print("LIVE_HARDWARE=False: no board connection was attempted.")
        print_offline_channel_expectation(repository)
        return None
    return connect_quick(repository, apply_rf_board=apply_rf_board)


def _grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
) -> tuple:
    x_values = np.unique(x)
    y_values = np.unique(y)
    result = np.full((y_values.size, x_values.size), np.nan)
    result[
        np.searchsorted(y_values, y),
        np.searchsorted(x_values, x),
    ] = values
    return x_values, y_values, result


def plot_data(data: ExperimentData, title: str):
    axes = list(data.axes)
    signals = data.signals
    if not axes and {
        "i_ground", "q_ground", "i_excited", "q_excited"
    } <= set(signals):
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        ax.scatter(
            signals["i_ground"],
            signals["q_ground"],
            s=5,
            alpha=0.35,
            label="ground",
        )
        ax.scatter(
            signals["i_excited"],
            signals["q_excited"],
            s=5,
            alpha=0.35,
            label="excited",
        )
        ax.set(xlabel="I", ylabel="Q", title=title)
        ax.axis("equal")
        ax.grid(alpha=0.25)
        ax.legend()
        return fig

    if not axes:
        return None
    if len(axes) == 1:
        x_name = axes[0]
        x = data.axes[x_name]
        if {"amplitude_ground", "amplitude_excited"} <= set(signals):
            fig, plots = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            plots[0].plot(x, signals["amplitude_ground"], label="ground")
            plots[0].plot(x, signals["amplitude_excited"], label="excited")
            separation = np.hypot(
                signals["i_excited"] - signals["i_ground"],
                signals["q_excited"] - signals["q_ground"],
            )
            plots[1].plot(x, separation)
            plots[0].legend()
            plots[0].set(ylabel="Amplitude")
            plots[1].set(ylabel="IQ separation")
        else:
            fig, plots = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            amplitude = signals.get("amplitude", np.abs(data.iq))
            phase = signals.get("phase", np.angle(data.iq))
            plots[0].plot(x, amplitude, ".-")
            plots[1].plot(x, np.unwrap(phase), ".-")
            plots[0].set(ylabel="Amplitude")
            plots[1].set(ylabel="Unwrapped phase (rad)")
        for plot in plots:
            plot.set(xlabel=x_name)
            plot.grid(alpha=0.25)
        fig.suptitle(title)
        return fig

    x_name, y_name = axes[:2]
    x, y = data.axes[x_name], data.axes[y_name]
    amplitude = signals.get("amplitude", np.abs(data.iq))
    phase = signals.get("phase", np.angle(data.iq))
    x_values, y_values, amplitude_grid = _grid(x, y, amplitude)
    _, _, phase_grid = _grid(x, y, phase)
    fig, plots = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    amp_mesh = plots[0].pcolormesh(
        x_values, y_values, amplitude_grid, shading="auto"
    )
    phase_mesh = plots[1].pcolormesh(
        x_values,
        y_values,
        np.unwrap(phase_grid, axis=1),
        shading="auto",
        cmap="twilight",
    )
    plots[0].set(xlabel=x_name, ylabel=y_name, title="Amplitude")
    plots[1].set(xlabel=x_name, ylabel=y_name, title="Phase")
    fig.colorbar(amp_mesh, ax=plots[0])
    fig.colorbar(phase_mesh, ax=plots[1])
    fig.suptitle(title)
    return fig


def _print_envelope_preflight(details: Optional[Mapping[str, Any]]) -> None:
    if not details:
        return
    if "program" in details:
        program = details.get("program", {})
        envelopes = program.get("envelopes", {}) if isinstance(program, Mapping) else {}
        required = sum(
            int(value.get("required_samples", 0))
            for value in envelopes.values()
            if isinstance(value, Mapping)
        )
        print(
            "Authored Mercator preflight: "
            f"{required} waveform samples budgeted across "
            f"{len(envelopes)} envelope term(s)."
        )
        for warning in details.get("warnings", ()):
            print(f"Mercator preflight warning: {warning}")
        return
    print(
        "Quick envelope preflight: "
        f"{details['required_samples']}/{details['available_samples']} "
        f"q-generator samples required; single-envelope nominal maximum "
        f"{details['maximum_single_envelope_us']:.6f} us."
    )


def run_experiment(
    project_root: Path,
    *,
    experiment: str,
    preset: str,
    overrides: Optional[Mapping[str, Any]] = None,
    run_options: Optional[Mapping[str, Any]] = None,
    title: Optional[str] = None,
    live_hardware: bool = False,
    fixed_z_gain: Optional[float] = None,
    flux_line: str = "z_rf",
    analyze: bool = True,
    show_plot: bool = True,
    seed: int = 7,
    backend: Any = None,
) -> CompletedRun:
    repository = load_repository(project_root)
    run_overrides = deepcopy(dict(overrides or {}))
    if fixed_z_gain is not None:
        run_overrides["z_gain"] = float(fixed_z_gain)
    connection = None
    flux = None
    if live_hardware:
        if backend is not None:
            raise ValueError("a custom backend cannot be supplied for live hardware")
        connection = connect_quick(repository)
        backend = connection.backend
        try:
            preflight_runner = ExperimentRunner(repository, backend)
            preflight_plan = preflight_runner.plan(
                experiment,
                preset,
                overrides=run_overrides,
                run_options=run_options,
                title=title,
            )
            validate = getattr(backend, "validate", None)
            preflight_details = (
                validate(preflight_plan.plan) if callable(validate) else None
            )
            _print_envelope_preflight(preflight_details)
        except BaseException:
            try:
                backend.close()
            except BaseException as cleanup_error:
                print(f"Backend preflight cleanup warning: {cleanup_error}")
            raise
        if fixed_z_gain is not None:
            requested_z = float(fixed_z_gain)
            line = repository.hardware.get("flux_lines", {}).get(flux_line, {})
            park_z = float(line.get("park", 0.0))
            reset_on_connect = bool(
                repository.hardware.get("qick", {}).get(
                    "reset_generators_on_connect",
                    True,
                )
            )
            if (
                reset_on_connect
                and np.isclose(requested_z, 0.0)
                and np.isclose(park_z, 0.0)
            ):
                print(
                    "Z gain is the zero park value; generator reset already "
                    "established zero, so the held-Z helper is skipped."
                )
            else:
                try:
                    resolved = repository.resolve(
                        preset,
                        experiment=experiment,
                        overrides=run_overrides,
                    )
                    flux = make_held_flux_controller(
                        connection,
                        repository,
                        resolved.expanded_parameters(),
                        line_name=flux_line,
                    )
                    flux.set(requested_z)
                except BaseException:
                    if flux is not None:
                        try:
                            flux.park()
                        except BaseException as cleanup_error:
                            print(
                                f"Held-Z setup cleanup warning: {cleanup_error}"
                            )
                    try:
                        backend.close()
                    except BaseException as cleanup_error:
                        print(f"Backend setup cleanup warning: {cleanup_error}")
                    raise
    else:
        print("SIMULATION: LIVE_HARDWARE=False")
        print_offline_channel_expectation(repository)
        backend = backend if backend is not None else SyntheticBackend(seed=seed)

    runner = ExperimentRunner(repository, backend, flux=flux)
    completed = runner.run(
        experiment,
        preset,
        overrides=run_overrides,
        run_options=run_options,
        title=title,
        analyze=analyze,
    )
    native_files = list(completed.data.metadata.get("native_files", []))
    print(f"{completed.status}: {completed.data.points} points")
    if native_files:
        print("Quick native files:")
        for path in native_files:
            print(f"  {path}")
    else:
        print("No v3 run directory was written (simulation/in-memory result).")
    if completed.analysis is not None:
        print("Analysis:", completed.analysis.as_dict())
    figure = plot_data(completed.data, title or preset)
    if figure is not None and show_plot:
        plt.show()
    return completed


def resonator_frequency_from_flux(
    repository: ConfigRepository,
    z_gain: Any,
) -> Any:
    try:
        record = repository.calibration["records"]["lookups"][
            "resonator_vs_flux"
        ]
        return frequency_from_record(record, z_gain)
    except (KeyError, TypeError, ValueError, ConfigError) as error:
        raise RuntimeError(
            str(error)
            if "outside accepted" in str(error)
            else (
                "calibration.yml has no accepted "
                "lookups.resonator_vs_flux cosine"
            )
        ) from error


def resolve_readout_frequency(
    project_root: Path,
    z_gain: float,
    *,
    use_accepted_fit: bool,
    fixed_frequency_mhz: float,
) -> float:
    """Choose and report a fitted or explicit readout frequency for one Z."""
    if use_accepted_fit:
        repository = load_repository(project_root)
        frequency = float(resonator_frequency_from_flux(repository, z_gain))
        print(
            f"Accepted resonator-vs-Z fit: Z={float(z_gain):+.6f} -> "
            f"r_freq={frequency:.6f} MHz"
        )
        return frequency
    frequency = float(fixed_frequency_mhz)
    if not np.isfinite(frequency):
        raise ValueError("fixed_frequency_mhz must be finite")
    print(
        f"Manual readout frequency: Z={float(z_gain):+.6f} -> "
        f"r_freq={frequency:.6f} MHz"
    )
    return frequency


def run_flux_sweep(
    project_root: Path,
    *,
    experiment: str,
    preset: str,
    flux_values: Sequence[float],
    overrides: Optional[Mapping[str, Any]] = None,
    run_options: Optional[Mapping[str, Any]] = None,
    title: Optional[str] = None,
    live_hardware: bool = False,
    flux_line: str = "z_rf",
    readout_frequency: Optional[Callable[[float], float]] = None,
    readout_metadata: Optional[Mapping[str, Any]] = None,
    analyze_rows: bool = False,
    show_plot: bool = True,
    seed: int = 7,
    backend: Any = None,
    before_row: Optional[Callable[[int, float], None]] = None,
    after_row: Optional[
        Callable[[int, float, Optional[CompletedRun]], None]
    ] = None,
) -> list:
    """Acquire a held-Z map and save one combined native Quick CSV/YML."""
    values = np.asarray(flux_values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("flux_values must be a finite, non-empty 1D sequence")
    readout_values = None
    if readout_frequency is not None:
        readout_values = np.asarray(
            [readout_frequency(float(value)) for value in values],
            dtype=float,
        )
        if readout_values.shape != values.shape or not np.all(
            np.isfinite(readout_values)
        ):
            raise ValueError(
                "readout_frequency must return one finite value per Z"
            )
    repository = load_repository(project_root)
    base_overrides = deepcopy(dict(overrides or {}))
    connection = None
    flux = None
    if live_hardware:
        if backend is not None:
            raise ValueError("a custom backend cannot be supplied for live hardware")
        connection = connect_quick(repository)
        backend = connection.backend
        try:
            initial_overrides = deepcopy(base_overrides)
            initial_overrides["z_gain"] = float(values[0])
            if readout_values is not None:
                initial_overrides["r_freq"] = float(readout_values[0])
            initial_runner = ExperimentRunner(repository, backend)
            initial = initial_runner.plan(
                experiment,
                preset,
                overrides=initial_overrides,
                run_options=run_options,
                title=title,
            )
            validate = getattr(backend, "validate", None)
            preflight_details = (
                validate(initial.plan) if callable(validate) else None
            )
            _print_envelope_preflight(preflight_details)
            flux = make_held_flux_controller(
                connection,
                repository,
                initial.config.expanded_parameters(),
                line_name=flux_line,
            )
        except BaseException:
            try:
                backend.close()
            except BaseException as cleanup_error:
                print(f"Backend setup cleanup warning: {cleanup_error}")
            raise
    else:
        print("SIMULATION: LIVE_HARDWARE=False")
        print_offline_channel_expectation(repository)
        backend = backend if backend is not None else SyntheticBackend(seed=seed)

    runner = ExperimentRunner(repository, backend, flux=flux)
    combined_title = title or f"{preset}_vs_Z"
    native_saver = None
    native_files = []
    original_data_path = getattr(backend, "data_path", None)
    if live_hardware and connection is not None and original_data_path:
        preview_overrides = deepcopy(base_overrides)
        preview_overrides["z_gain"] = float(values[0])
        if readout_values is not None:
            preview_overrides["r_freq"] = float(readout_values[0])
        preview = runner.plan(
            experiment,
            preset,
            overrides=preview_overrides,
            run_options=run_options,
            title=combined_title,
        )
        if len(preview.plan.axes) != 1:
            raise ValueError(
                "run_flux_sweep requires exactly one inner Quick sweep axis"
            )
        inner_name = preview.plan.axes[0]
        axis_labels = {
            "r_freq": ("Frequency", "MHz"),
            "q_freq": ("Qubit Frequency", "MHz"),
        }
        independent = [
            ("Z Gain", "a.u."),
            axis_labels.get(
                inner_name,
                (inner_name, preview.plan.axis_units.get(inner_name, "")),
            ),
        ]
        signal_labels = {
            "population": "Population",
            "amplitude": "Amplitude",
            "phase": "Phase",
            "i": "I",
            "q": "Q",
        }
        dependent = []
        if readout_values is not None:
            dependent.append(("Readout Frequency", "MHz"))
        dependent.extend(
            (
                signal_labels.get(name, name),
                preview.plan.signal_units.get(name, ""),
            )
            for name in preview.plan.signal_names
        )
        parameters = to_builtin(dict(preview.plan.variables))
        parameters["preset"] = preset
        parameters["z_gain_sweep"] = values.tolist()
        if readout_values is not None:
            parameters["r_freq_by_z"] = readout_values.tolist()
        if readout_metadata is not None:
            parameters["readout_frequency_calibration"] = to_builtin(
                readout_metadata
            )
        parameters[f"{inner_name}_sweep"] = to_builtin(
            preview.plan.variables[inner_name]
        )
        native_saver = connection.quick.Saver(
            combined_title,
            original_data_path,
            indep_params=independent,
            dep_params=dependent,
            params=parameters,
        )
        native_files = [
            native_saver.file_name + ".csv",
            native_saver.file_name + ".yml",
        ]
        # Each row is still acquired by Quick, but only this combined Saver
        # writes data so a flux map remains one CSV/YML pair.
        backend.data_path = None

    completed_rows = []
    try:
        if live_hardware and connection is not None:
            progress = connection.quick.Sweep(
                {"z_gain": float(values[0])},
                {"z_gain": values},
                progressBar=True,
            )
            row_values = (float(state["z_gain"]) for state in progress)
        else:
            try:
                from tqdm.auto import tqdm

                row_values = tqdm(values, desc="Z sweep", unit="row")
            except ImportError:
                row_values = values

        for row_index, value in enumerate(row_values):
            value = float(value)
            if before_row is not None:
                before_row(row_index, value)
            row_overrides = deepcopy(base_overrides)
            row_overrides["z_gain"] = value
            row_readout = None
            if readout_values is not None:
                row_readout = float(readout_values[row_index])
                row_overrides["r_freq"] = row_readout
            if flux is not None:
                flux.set(value)
            completed = None
            try:
                completed = runner.run(
                    experiment,
                    preset,
                    overrides=row_overrides,
                    run_options=run_options,
                    title=combined_title,
                    analyze=analyze_rows,
                    close_backend=False,
                    park_flux_on_exit=False,
                )
            finally:
                if after_row is not None:
                    after_row(row_index, value, completed)
            completed_rows.append(completed)
            if native_saver is not None:
                inner_axis = next(iter(completed.data.axes.values()))
                columns = [
                    np.full(np.asarray(inner_axis).shape, value, dtype=float),
                    np.asarray(inner_axis, dtype=float),
                ]
                if row_readout is not None:
                    columns.append(
                        np.full(
                            np.asarray(inner_axis).shape,
                            row_readout,
                            dtype=float,
                        )
                    )
                columns.extend(
                    np.asarray(signal, dtype=float)
                    for signal in completed.data.signals.values()
                )
                native_saver.write_data(np.column_stack(columns))
    finally:
        if native_saver is not None and native_saver.has_data:
            native_saver.write_yml()
        if original_data_path is not None:
            backend.data_path = original_data_path
        if flux is not None:
            flux.park()
        backend.close()

    if native_files:
        for row in completed_rows:
            row.data.metadata["native_files"] = list(native_files)
        print("Quick native flux-map files:")
        for path in native_files:
            print(f"  {path}")

    if completed_rows:
        inner_name = next(iter(completed_rows[0].data.axes))
        inner = np.asarray(completed_rows[0].data.axes[inner_name])
        matrix = []
        for row in completed_rows:
            signals = row.data.signals
            matrix.append(
                np.asarray(
                    signals.get("amplitude", np.abs(row.data.iq)),
                    dtype=float,
                )
            )
        figure, plot = plt.subplots(figsize=(8, 5), constrained_layout=True)
        mesh = plot.pcolormesh(
            inner,
            values,
            np.asarray(matrix),
            shading="auto",
        )
        plot.set(
            xlabel=inner_name,
            ylabel="z_gain",
            title=combined_title,
        )
        figure.colorbar(mesh, ax=plot)
        if show_plot:
            plt.show()
    return completed_rows
