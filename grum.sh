#!/bin/bash

if ! [[ -x 'tests/grum.py' ]]; then
    echo "must be run from an assignment folder with a tests/ subfolder." >&2
    exit 1;
fi

f="$(readlink "$0")"
d="$(dirname "$f")"
grep -v 'sys.path\[0:0\]=' tests/grum.py | PYTHONPATH="$d:$PYTHONPATH" python - "$@"
#PYTHONPATH="$d:$PYTHONPATH" python2 tests/grum.py "$@"
