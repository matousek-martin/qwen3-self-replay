"""Reload the controller at an idle boundary without interrupting an API request."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'lab/experiments/teacher-generation/isolated-pipeline-20261025'


def main():
    while True:
        state = json.loads((STATE / 'pipeline.json').read_text())
        phase = state['phase']
        if phase.startswith('complete'):
            print('Pipeline already finished; future controllers use $200.', flush=True)
            return
        if phase in ('queued_gpu', 'queued_review_budget'):
            # Freeze the local control cgroup before checking its phase again.
            # Workers on rented GPUs are separate and are never signalled.
            unit = 'thinkingcap-isolated-teacher-pipeline.service'
            subprocess.run(['systemctl', 'freeze', unit], check=True)
            try:
                current = json.loads((STATE / 'pipeline.json').read_text())
                if current['phase'] != phase:
                    continue
                # Stop while frozen so no new API dispatch/allocation can race
                # between the verified idle boundary and the restart.
                # At this idle boundary only the main Python process needs
                # signalling. Some systemd versions return an error for
                # --kill-whom=all after successfully killing that process when
                # there are no auxiliary processes, skipping the restart.
                subprocess.run(['systemctl', 'kill', '--kill-whom=main', '--signal=SIGKILL', unit], check=True)
            finally:
                subprocess.run(['systemctl', 'thaw', unit], check=False, capture_output=True)
            subprocess.run(['systemctl', 'restart', unit], check=True)
            print('Reloaded idle teacher controller with full $200 allowance.', flush=True)
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
