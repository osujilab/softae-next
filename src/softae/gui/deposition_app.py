"""Standalone deposition digital-twin app.

Launch with::

    python -m softae.gui.deposition_app
    # or
    softae-deposition
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from softae.config import loader as cfg
from softae.core.formulation import ChemicalCatalog, SolutionCatalog
from softae.gui.widgets.deposition_panel import DepositionPanel
from softae.tools import use_utf8_console


def load_catalogs(config: dict | None) -> tuple[ChemicalCatalog, SolutionCatalog, str]:
    """Load chemical/solution catalogs from ``{data_root}/chemicals.csv|solutions.csv``.

    ``config`` is the parsed softae_config.toml dict (or None when no config was
    found).  Never raises: any missing config/section degrades to empty
    catalog(s), falling back per-file from ``{data_root}`` to the repo-root
    convention (``./chemicals.csv`` / ``./solutions.csv``).  A file that EXISTS
    but fails to parse is caught and surfaced in the status message (the catalog
    degrades to empty) rather than being silently swallowed; the returned status
    message says what was (not) loaded.

    ``data_root`` resolution: an explicit ``config["paths"]["data_root"]`` wins
    (used verbatim, preserving testability); otherwise the config-anchored
    ``loader.data_root()`` is used so a launch from any directory still finds the
    canonical catalogs.  A final CWD-relative ``Path(filename)`` fallback per file
    keeps the repo-root convention working.
    """
    paths_section = (config or {}).get("paths", {})
    if not isinstance(paths_section, dict):
        paths_section = {}
    if "data_root" in paths_section:
        data_root = str(paths_section.get("data_root", "./data"))
    elif config is None:
        # No config file at all → CWD-relative default (matches data_root()'s own
        # no-config fallback); avoids anchoring to an unrelated repo-root config.
        data_root = "./data"
    else:
        try:
            data_root = str(cfg.data_root())
        except Exception:
            data_root = "./data"

    def _load_one(catalog_cls, filename: str):
        """Return ``(catalog, source_path, parse_error)``.

        ``source_path`` is the loaded file (``None`` if none was found);
        ``parse_error`` is a short reason string when an existing file was
        found but could not be parsed (``None`` otherwise).  Never raises.
        """
        parse_error = None
        for candidate in (Path(data_root) / filename, Path(filename)):
            if not candidate.is_file():
                continue
            try:
                return catalog_cls.load_csv(candidate), candidate, None
            except Exception as exc:  # noqa: BLE001 - never raise; report instead
                if parse_error is None:
                    parse_error = (candidate, f"{type(exc).__name__}: {exc}")
        if parse_error is not None:
            return catalog_cls(), parse_error[0], parse_error[1]
        return catalog_cls(), None, None

    chem_cat, chem_src, chem_err = _load_one(ChemicalCatalog, "chemicals.csv")
    sol_cat, sol_src, sol_err = _load_one(SolutionCatalog, "solutions.csv")

    if chem_err is None and sol_err is None:
        if chem_src is not None and sol_src is not None:
            status = (
                f"Loaded {len(chem_cat)} chemicals, {len(sol_cat)} solutions "
                f"from {chem_src.parent}"
            )
        elif chem_src is not None:
            status = (
                f"Loaded {len(chem_cat)} chemicals from {chem_src.parent}; "
                "no solutions.csv found — solutions empty."
            )
        elif sol_src is not None:
            status = (
                f"Loaded {len(sol_cat)} solutions from {sol_src.parent}; "
                "no chemicals.csv found — chemicals empty."
            )
        else:
            status = (
                f"No catalogs found under {data_root} — starting empty. "
                "Use FormulationPanel to create them."
            )
    else:
        def _phrase(label: str, cat, src, err) -> str:
            if err is not None:
                return (
                    f"{label.capitalize()} file at {src} could not be read "
                    f"({err}) — {label} empty."
                )
            if src is not None:
                return f"Loaded {len(cat)} {label} from {src.parent}."
            return f"No {label}.csv found — {label} empty."

        status = " ".join((
            _phrase("chemicals", chem_cat, chem_src, chem_err),
            _phrase("solutions", sol_cat, sol_src, sol_err),
        ))
    return chem_cat, sol_cat, status


def _reload_into(panel: DepositionPanel, config: dict | None) -> None:
    """Re-read catalogs from ``data_root`` and swap them into an open panel."""
    chem_cat, sol_cat, status = load_catalogs(config)
    panel.set_catalogs(chem_cat, sol_cat)
    panel.show_status(status)


def _open_editor(panel: DepositionPanel, config: dict | None) -> None:
    """Open the FormulationPanel editor modally; live-reload on each save."""
    from softae.gui.widgets.formulation_panel import FormulationPanel

    dlg = FormulationPanel(parent=panel)
    dlg.catalogs_changed.connect(lambda: _reload_into(panel, config))
    dlg.exec()
    _reload_into(panel, config)  # final safety reload after the dialog closes


def main() -> None:
    """CLI entry point for the standalone deposition twin GUI."""
    use_utf8_console()
    try:
        config = cfg.load()
    except FileNotFoundError:
        config = None

    chem_cat, sol_cat, status = load_catalogs(config)

    app = QApplication(sys.argv)
    app.setApplicationName("SoftAE Deposition")
    app.setOrganizationName("OsujiLab")

    panel = DepositionPanel(chem_cat, sol_cat)
    panel.setWindowTitle("Deposition Digital Twin")
    panel.show_status(status)
    panel.manage_catalogs_requested.connect(lambda: _open_editor(panel, config))
    panel.reload_catalogs_requested.connect(lambda: _reload_into(panel, config))
    panel.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
