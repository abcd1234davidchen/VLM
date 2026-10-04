"""Record periodic summaries without holding an interactive training session."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=1200)
    args = parser.parse_args()
    if args.interval < 60:
        parser.error('Summary interval must be at least 60 seconds')
    root = args.artifacts_root
    while True:
        arm = root / 'semantic/pilot/seed42'
        progress = read(arm / 'progress.json')
        selection = read(arm / 'selection.json')
        percent = (100 * progress['samples'] / progress['total']
                   if progress.get('total') and 'samples' in progress else None)
        snapshot = {'timestamp': datetime.now(timezone.utc).isoformat(),
                    'interval_seconds': args.interval, 'progress': progress,
                    'display': f"1-{progress.get('epoch', '?')}-{round(percent) if percent is not None else '?'}",
                    'best': {view: {key: values.get(key) for key in ('epoch', 'raw_accuracy', 'raw_macro_f1', 'correct')}
                             for view, values in selection.get('best', {}).items() if values},
                    'workflow_control': read(root / 'workflow_control.json'),
                    'workflow_status': read(root / 'dispatch_status.json'),
                    'acceleration_progress': read(root.parent / 'acceleration_parallel_v1/progress.json'),
                    'anchor_cache_progress': read(root.parent / 'shared_anchor_cache_v1/progress.json'),
                    'anchor_cache_dispatch': read(root.parent / 'shared_anchor_cache_v1/dispatch_status.json'),
                    'test_evaluated': False}
        temporary = root / 'monitor_20min.tmp'
        temporary.write_text(json.dumps(snapshot, indent=2))
        temporary.replace(root / 'monitor_20min_status.json')
        with (root / 'monitor_20min.jsonl').open('a') as stream:
            stream.write(json.dumps(snapshot) + '\n')
        print(json.dumps(snapshot), flush=True)
        # Keep monitoring after completion so terminal state remains available.
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
