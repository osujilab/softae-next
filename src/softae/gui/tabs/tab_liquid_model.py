"""Dedicated liquid-correction model editor tab."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager


class LiquidModelTab(QWidget):
    """Editor for liquid-handling correction parameters."""

    def __init__(
        self,
        manager: InstrumentManager | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        try:
            from softae.config.loader import liquid_handling_config

            liq_cfg = liquid_handling_config()
        except Exception:
            liq_cfg = {
                "enabled": False,
                "beta": 0.30,
                "eta_ref_mpas": 1.0,
                "alpha_growth_per_run": 0.0,
                "line": {
                    "0": {
                        "cracking_kpa_per_valve": 8.0,
                        "compliance_uL_per_kpa": 0.55,
                        "alpha_base": 0.20,
                        "viscosity_mpas": 1.0,
                    },
                    "1": {
                        "cracking_kpa_per_valve": 8.0,
                        "compliance_uL_per_kpa": 0.55,
                        "alpha_base": 0.20,
                        "viscosity_mpas": 1.0,
                    },
                    "2": {
                        "cracking_kpa_per_valve": 8.0,
                        "compliance_uL_per_kpa": 0.55,
                        "alpha_base": 0.20,
                        "viscosity_mpas": 1.0,
                    },
                },
            }

        try:
            from softae.config.loader import piezo_config

            piezo_cfg = piezo_config()
        except Exception:
            piezo_cfg = {
                "enabled": False,
                "channel": "A",
                "frequency_hz": 500,
                "sweep_on_s": 2.0,
                "sweep_rest_s": 3.0,
                "liquid_events": {
                    "enabled": False,
                    "settings_source": "manual_profile",
                    "channel_a": True,
                    "frequency_hz": 500,
                    "sweep_on_s": 2.0,
                    "sweep_rest_s": 3.0,
                },
            }

        sys_box = QGroupBox("System Parameters")
        sys_grid = QGridLayout(sys_box)

        self._chk_liq_enabled = QCheckBox("Liquid correction enabled")
        self._chk_liq_enabled.setChecked(bool(liq_cfg.get("enabled", False)))
        sys_grid.addWidget(self._chk_liq_enabled, 0, 0, 1, 2)

        sys_grid.addWidget(QLabel("beta:"), 1, 0)
        self._spin_liq_beta = QDoubleSpinBox()
        self._spin_liq_beta.setRange(0.0, 5.0)
        self._spin_liq_beta.setDecimals(3)
        self._spin_liq_beta.setSingleStep(0.01)
        self._spin_liq_beta.setValue(float(liq_cfg.get("beta", 0.30)))
        sys_grid.addWidget(self._spin_liq_beta, 1, 1)

        sys_grid.addWidget(QLabel("eta_ref (mPa*s):"), 2, 0)
        self._spin_liq_eta_ref = QDoubleSpinBox()
        self._spin_liq_eta_ref.setRange(0.01, 10000.0)
        self._spin_liq_eta_ref.setDecimals(3)
        self._spin_liq_eta_ref.setSingleStep(0.1)
        self._spin_liq_eta_ref.setValue(float(liq_cfg.get("eta_ref_mpas", 1.0)))
        sys_grid.addWidget(self._spin_liq_eta_ref, 2, 1)

        sys_grid.addWidget(QLabel("alpha growth/run:"), 3, 0)
        self._spin_liq_alpha_growth = QDoubleSpinBox()
        self._spin_liq_alpha_growth.setRange(0.0, 1.0)
        self._spin_liq_alpha_growth.setDecimals(3)
        self._spin_liq_alpha_growth.setSingleStep(0.01)
        self._spin_liq_alpha_growth.setValue(float(liq_cfg.get("alpha_growth_per_run", 0.0)))
        sys_grid.addWidget(self._spin_liq_alpha_growth, 3, 1)

        root.addWidget(sys_box)

        self._line_liq_widgets: dict[int, dict[str, QDoubleSpinBox | QLabel]] = {}
        line_cfgs = liq_cfg.get("line", {})
        if not isinstance(line_cfgs, dict):
            line_cfgs = {}

        for line_id in (0, 1, 2):
            row_box = QGroupBox(f"Line {line_id}")
            row_layout = QGridLayout(row_box)
            cfg = line_cfgs.get(str(line_id), line_cfgs.get(line_id, {}))
            if not isinstance(cfg, dict):
                cfg = {}

            def _spin(
                value: float,
                minimum: float,
                maximum: float,
                decimals: int = 3,
                step: float = 0.1,
            ) -> QDoubleSpinBox:
                spin = QDoubleSpinBox()
                spin.setRange(minimum, maximum)
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
                spin.setValue(value)
                return spin

            row_layout.addWidget(QLabel("Cracking (kPa/valve):"), 0, 0)
            spin_crack = _spin(float(cfg.get("cracking_kpa_per_valve", 8.0)), 0.0, 1000.0)
            row_layout.addWidget(spin_crack, 0, 1)

            row_layout.addWidget(QLabel("Compliance (uL/kPa):"), 1, 0)
            spin_comp = _spin(
                float(cfg.get("compliance_uL_per_kpa", 0.55)),
                0.0,
                10.0,
                decimals=4,
                step=0.01,
            )
            row_layout.addWidget(spin_comp, 1, 1)

            row_layout.addWidget(QLabel("Alpha base:"), 2, 0)
            spin_alpha = _spin(float(cfg.get("alpha_base", 0.20)), 0.0, 1.0, decimals=3, step=0.01)
            row_layout.addWidget(spin_alpha, 2, 1)

            row_layout.addWidget(QLabel("Viscosity (mPa*s):"), 3, 0)
            spin_visc = _spin(float(cfg.get("viscosity_mpas", 1.0)), 0.01, 10000.0, decimals=3, step=0.1)
            row_layout.addWidget(spin_visc, 3, 1)

            row_layout.addWidget(QLabel("Estimated prime:"), 4, 0)
            lbl_prime = QLabel("-- uL")
            lbl_prime.setStyleSheet("font-weight: bold;")
            row_layout.addWidget(lbl_prime, 4, 1)

            self._line_liq_widgets[line_id] = {
                "cracking": spin_crack,
                "compliance": spin_comp,
                "alpha": spin_alpha,
                "viscosity": spin_visc,
                "prime": lbl_prime,
            }
            for spin in (spin_crack, spin_comp, spin_alpha, spin_visc):
                spin.valueChanged.connect(self._refresh_liquid_prime_labels)

            root.addWidget(row_box)

        piezo_box = QGroupBox("Piezo Event Settings")
        piezo_grid = QGridLayout(piezo_box)
        piezo_events = piezo_cfg.get("liquid_events", {})
        if not isinstance(piezo_events, dict):
            piezo_events = {}

        self._chk_piezo_events_enabled = QCheckBox("Enable piezo during liquid-handling events")
        self._chk_piezo_events_enabled.setChecked(bool(piezo_events.get("enabled", False)))
        piezo_grid.addWidget(self._chk_piezo_events_enabled, 0, 0, 1, 2)

        piezo_grid.addWidget(QLabel("Settings source:"), 1, 0)
        self._combo_piezo_source = QComboBox()
        self._combo_piezo_source.addItem("manual_profile")
        self._combo_piezo_source.addItem("liquid_event_profile")
        source = str(piezo_events.get("settings_source", "manual_profile"))
        idx = self._combo_piezo_source.findText(source)
        self._combo_piezo_source.setCurrentIndex(idx if idx >= 0 else 0)
        self._combo_piezo_source.currentTextChanged.connect(self._refresh_piezo_event_inputs_enabled)
        piezo_grid.addWidget(self._combo_piezo_source, 1, 1)

        self._chk_piezo_channel_a = QCheckBox("Use channel A for events")
        self._chk_piezo_channel_a.setChecked(bool(piezo_events.get("channel_a", True)))
        piezo_grid.addWidget(self._chk_piezo_channel_a, 2, 0, 1, 2)

        piezo_grid.addWidget(QLabel("Event freq (Hz):"), 3, 0)
        self._spin_piezo_event_freq = QSpinBox()
        self._spin_piezo_event_freq.setRange(10, 5000)
        self._spin_piezo_event_freq.setValue(int(piezo_events.get("frequency_hz", 500)))
        piezo_grid.addWidget(self._spin_piezo_event_freq, 3, 1)

        piezo_grid.addWidget(QLabel("Event ON (s):"), 4, 0)
        self._spin_piezo_event_on_s = QDoubleSpinBox()
        self._spin_piezo_event_on_s.setRange(0.01, 120.0)
        self._spin_piezo_event_on_s.setDecimals(3)
        self._spin_piezo_event_on_s.setSingleStep(0.1)
        self._spin_piezo_event_on_s.setValue(float(piezo_events.get("sweep_on_s", 2.0)))
        piezo_grid.addWidget(self._spin_piezo_event_on_s, 4, 1)

        piezo_grid.addWidget(QLabel("Event REST (s):"), 5, 0)
        self._spin_piezo_event_rest_s = QDoubleSpinBox()
        self._spin_piezo_event_rest_s.setRange(0.01, 120.0)
        self._spin_piezo_event_rest_s.setDecimals(3)
        self._spin_piezo_event_rest_s.setSingleStep(0.1)
        self._spin_piezo_event_rest_s.setValue(float(piezo_events.get("sweep_rest_s", 3.0)))
        piezo_grid.addWidget(self._spin_piezo_event_rest_s, 5, 1)

        self._piezo_event_profile_inputs = [
            self._spin_piezo_event_freq,
            self._spin_piezo_event_on_s,
            self._spin_piezo_event_rest_s,
        ]
        root.addWidget(piezo_box)
        self._refresh_piezo_event_inputs_enabled()

        self._btn_apply_liq = QPushButton("Apply + Save")
        self._btn_apply_liq.clicked.connect(self._on_apply_liquid_model)
        root.addWidget(self._btn_apply_liq)

        self._lbl_liq_status = QLabel("")
        self._lbl_liq_status.setWordWrap(True)
        root.addWidget(self._lbl_liq_status)

        root.addStretch()
        self._refresh_liquid_prime_labels()

    def _liquid_model_section_from_ui(self) -> dict[str, Any]:
        line_section: dict[str, dict[str, float]] = {}
        for line_id, widgets in self._line_liq_widgets.items():
            line_section[str(line_id)] = {
                "cracking_kpa_per_valve": float(widgets["cracking"].value()),
                "compliance_uL_per_kpa": float(widgets["compliance"].value()),
                "alpha_base": float(widgets["alpha"].value()),
                "viscosity_mpas": float(widgets["viscosity"].value()),
            }
        return {
            "enabled": bool(self._chk_liq_enabled.isChecked()),
            "beta": float(self._spin_liq_beta.value()),
            "eta_ref_mpas": float(self._spin_liq_eta_ref.value()),
            "alpha_growth_per_run": float(self._spin_liq_alpha_growth.value()),
            "line": line_section,
        }

    def _piezo_section_from_ui(self) -> dict[str, Any]:
        return {
            "liquid_events": {
                "enabled": bool(self._chk_piezo_events_enabled.isChecked()),
                "settings_source": self._combo_piezo_source.currentText(),
                "channel_a": bool(self._chk_piezo_channel_a.isChecked()),
                "frequency_hz": int(self._spin_piezo_event_freq.value()),
                "sweep_on_s": float(self._spin_piezo_event_on_s.value()),
                "sweep_rest_s": float(self._spin_piezo_event_rest_s.value()),
            }
        }

    def _refresh_piezo_event_inputs_enabled(self) -> None:
        event_profile = self._combo_piezo_source.currentText() == "liquid_event_profile"
        for widget in self._piezo_event_profile_inputs:
            widget.setEnabled(event_profile)

    def _refresh_liquid_prime_labels(self) -> None:
        try:
            from softae.core.liquid_handling import (
                LiquidHandlingCorrector,
                LinePhysicsConfig,
                SystemPhysicsConfig,
            )

            sys_cfg = SystemPhysicsConfig(
                beta=float(self._spin_liq_beta.value()),
                eta_ref_mpas=float(self._spin_liq_eta_ref.value()),
                alpha_growth_per_run=float(self._spin_liq_alpha_growth.value()),
            )
            corrector = LiquidHandlingCorrector()
            for line_id, widgets in self._line_liq_widgets.items():
                line_cfg = LinePhysicsConfig(
                    line_id=int(line_id),
                    cracking_kpa_per_valve=float(widgets["cracking"].value()),
                    compliance_uL_per_kpa=float(widgets["compliance"].value()),
                    alpha_base=float(widgets["alpha"].value()),
                    viscosity_mpas=float(widgets["viscosity"].value()),
                )
                widgets["prime"].setText(f"{corrector.prime_volume(line_cfg, sys_cfg):.2f} uL")
        except Exception as exc:
            self._lbl_liq_status.setText(f"Prime estimate error: {exc}")

    def _on_apply_liquid_model(self) -> None:
        try:
            from softae.config.loader import save_liquid_handling_config, save_piezo_config

            section = self._liquid_model_section_from_ui()
            piezo_section = self._piezo_section_from_ui()
            save_liquid_handling_config(section)
            save_piezo_config(piezo_section)
            self._lbl_liq_status.setText("Saved liquid correction and piezo event settings.")
        except Exception as exc:
            QMessageBox.warning(self, "Liquid Model Error", str(exc))
        finally:
            self._refresh_liquid_prime_labels()
