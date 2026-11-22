# Christus Autonomy — Phase 1 desktop control

This package turns the desktop from a scripted experiment into a **persistent,
permissioned control substrate**:

- continuous foreground/window observation;
- append-only experience log and novelty/transition model;
- real Windows mouse/keyboard input through `SendInput`;
- allowlisted applications and blocked sensitive titles;
- per-minute action budget;
- emergency stop file and global hotkey.

It is deliberately **not** an unrestricted administrator. The default policy
allows common applications only and blocks password/login/banking/system surfaces.
Edit `permissions.json` to add applications.

## Commands

```powershell
# Observe only, persistent
python -m christus_autonomy.cli start --mode observe

# Enable the action-queue drain (still allowlist/safety-checked)
python -m christus_autonomy.cli start --mode control

# One action, immediately (dry-run first is recommended)
python -m christus_autonomy.cli act --dry-run move 500 300
python -m christus_autonomy.cli act click 500 300
python -m christus_autonomy.cli act type "hello"
python -m christus_autonomy.cli act hotkey ctrl+s
python -m christus_autonomy.cli act activate Notepad
python -m christus_autonomy.cli act launch notepad.exe

# Status / stop / clear
python -m christus_autonomy.cli status
python -m christus_autonomy.cli stop
python -m christus_autonomy.cli clear-stop
```

Emergency stop: **Ctrl+Alt+Shift+Q**.

## What is and is not implemented

Implemented:

- Windows `SendInput` mouse/keyboard actuator;
- foreground/window perception;
- permission broker, audit log, rate limit, emergency stop;
- persistent observer and novelty memory;
- action queue for a future Christus policy.

Not yet implemented:

- online gradient updates of the core;
- autonomous goal generation;
- a learned policy that decides arbitrary desktop actions;
- administrator-level operations;
- multi-monitor coordinate calibration beyond virtual-screen coordinates.

The supervisor does not invent clicks. It drains `~/.christus_autonomy/actions.jsonl`
only in `control` mode. This is the integration point for a Christus policy.

## Safety

- Default allowlist only; no admin.
- Blocked titles include password, login, bank, payment, wallet, settings,
  registry and task manager.
- Blocked shortcuts include `Alt+F4`, `Win+R`, `Win+X`, `Ctrl+Shift+Esc` and
  `Ctrl+Alt+Delete`.
- `STOP` blocks every input action until `clear-stop`.
- Every action is written to `~/.christus_autonomy/audit.jsonl`.

Do not add applications that expose sensitive credentials. Do not run as
administrator.
