"""Readers for the CSV logs written by `utils.log_utils.CsvLogger`.

CsvLogger writes values with `str()`, so array-valued metrics (e.g. `training/actor/std` of the FB
agents) become multi-line numpy reprs and the file is not valid CSV. Numpy reprs contain no commas,
so joining physical lines while inside `[...]` restores one logical line per logged step.
"""
import csv
import io
import math


def _logical_lines(text):
    depth, buf = 0, []
    for line in text.splitlines():
        buf.append(line)
        depth += line.count('[') - line.count(']')
        if depth <= 0:
            yield ' '.join(buf)
            depth, buf = 0, []
    if buf:
        yield ' '.join(buf)


def read_rows(path):
    """List of {column: string} dicts, one per logged step."""
    with open(path) as f:
        text = f.read()
    return list(csv.DictReader(io.StringIO('\n'.join(_logical_lines(text)))))


def has_nan(value):
    """True if a cell is NaN, or an array repr that contains a NaN."""
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return isinstance(value, str) and 'nan' in value.replace('[', ' ').replace(']', ' ').split()
