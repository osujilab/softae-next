<#
.SYNOPSIS
    The equilibration characterization run, as one saved invocation.

.DESCRIPTION
    `plan` and `run` are separate processes sharing no state, so any flag not
    repeated on `run` silently reverts to its default. That cost a real run on
    2026-08-10: --preset fell back to Standard (40.7 s/channel against Quick's
    measured 10.47) and the electrode geometry was dropped whole, so every sigma
    in the run was NULL while every log line reported success.

    The tool now closes that itself. `plan --save` writes the FULLY RESOLVED
    design -- every value the run will use, including the defaults nobody typed
    -- and `run --from-plan` executes exactly that file. So this script no longer
    has to hold the design in sync between two modes by construction: BOTH modes
    write the plan from $Design below, and -Execute runs the file that was just
    written. The artifact is the contract, and it is timestamped on disk.

    Default is plan. Nothing is opened and nothing is heated without -Execute.

.PARAMETER Plan
    Where the resolved design is written. Overwritten on every invocation, so the
    file on disk is always the design in this script, never a stale one.

.EXAMPLE
    .\scripts\equilibration_run.ps1
    Prints the design, the budget, and what will refuse, and saves the plan.
    Touches no hardware.

.EXAMPLE
    .\scripts\equilibration_run.ps1 -Execute
    Re-saves the plan, then drives the chamber from it. Prompts for confirmation
    before any heat.
#>
param(
    [switch]$Execute,
    [string]$Plan
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'

if (-not $Plan) { $Plan = Join-Path $repo 'equilibration_plan.toml' }

# `softae-equilibration` is declared in pyproject.toml [project.scripts] but is
# NOT installed in this venv, so the console-script name does not resolve. The
# module form always does, and does not depend on a reinstall.
$cli = @('-m', 'softae.tools.equilibration')

# ---- The design. Declared once; saved to $Plan, executed from $Plan. --------
$Design = @(
    # Films are on channels 1-16; 4-7 are deliberately excluded from this run.
    '--channels', '1-3,8-16'

    # Measured on this rig: Quick 10.47 s/channel, Standard 40.85, Extended 115.2.
    # The cost is dominated by the low-frequency tail, not the point count.
    '--preset', 'Quick'

    # 12 channels x 10.47 s = 126 s, so a 240 s period is actually achievable.
    # At Standard the same round costs 488 s and the period cannot be honoured.
    '--round-period-s', '240'

    # [pcb.SoftAE_IDE_EIS] / [pcb.SoftAE_EIS_4Stripe] both carry L = w = 0.2 cm.
    # All THREE are required together -- sigma = L/(R*t*w) needs every term. The
    # CLI now REFUSES a partial geometry naming the missing flags, rather than
    # dropping the lot silently as it did through the 2026-08-10 run.
    '--electrode-l-cm', '0.2'
    '--electrode-w-cm', '0.2'
    '--electrode-t-cm', '0.02'        # 200 um dry target

    # Says the thickness above is a digital-twin target, NOT a measurement. This
    # board casts sessile (no wells), so nothing on it predicts a footprint.
    '--thickness-method', 'target'

    # Plan from the bench figure rather than preflight's model, which is ~10x low
    # (EIS_CYCLES_PER_POINT = 3.0 against a fitted ~29).
    '--measured-per-channel-s', '10.47'

    # ---- How long each setpoint is held --------------------------------------
    # --rounds is now a CEILING. The 2026-08-11 run held all eight setpoints for
    # the full 15 rounds and only the first needed it: the sigma swing was
    # 1600-2800% at S0, 57-1370% at S1, then 0.5-8.5% and 0.8-3.1% -- flat inside
    # a 5.98% measured noise floor. Seven setpoints x 45 min re-measured a
    # settled number.
    '--rounds', '15'

    # 10% clears that noise floor (22 of 96 series scattered above 20%, so a 2%
    # tolerance is unsatisfiable and the tool already refuses it for most series).
    # Three consecutive rounds inside the band; at least three channels must
    # carry usable evidence -- a NULL sigma or an R1 railed on the simpleSalt
    # 100 ohm bound is NOT evidence, however constant it looks. 325 of 1440 fits
    # in that run railed, reporting sigma = 0.5 S/cm with success = 1.
    '--settle', 'on'
    '--settle-tol-rel', '0.10'
    '--settle-n-rounds', '3'
    '--settle-min-channels', '3'

    # 1500 s ~ 3 tau at the first setpoint (tau = 425-575 s measured, films
    # drying from ambient to 15 %RH). Every later setpoint gets 600 s: the films
    # are dry, but the chamber still has to re-establish RH -- and it cannot
    # reach 15 %RH at 65 or 85 C at all, which is why the criterion is on SIGMA
    # and not on the RH process value.
    '--min-hold-first-s', '1500'
    '--min-hold-s', '600'
)

& $python @cli 'plan' @Design '--save' $Plan
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Execute) {
    # No design flags here on purpose: everything the run uses comes from $Plan,
    # and anything typed alongside --from-plan would print as an override diff.
    & $python @cli 'run' '--from-plan' $Plan '--execute'
}

exit $LASTEXITCODE
