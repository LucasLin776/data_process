"""Deterministic smoke and regression checks; NOT field performance evaluation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.stats import kurtosis

from detect import extract, fit_model, scores, decision


def run(*args):
    proc = subprocess.run([sys.executable, str(Path(__file__).with_name('detect.py')),
                           *map(str, args)], capture_output=True, text=True, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(proc.stdout + proc.stderr)


def main():
    rng = np.random.default_rng(42)
    sr = 8000
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        data = root/'wav'
        data.mkdir()
        for i in range(154):
            x = rng.normal(0, .02, sr*2)
            if i >= 120:
                for at in [4000, 8000, 12000]:
                    t = np.arange(120)/sr
                    x[at:at+120] += .6*np.exp(-t*400)*np.cos(2*np.pi*1700*t)
            wavfile.write(data/f'{"normal" if i<120 else "impact"}_{i:03d}.wav', sr, x.astype('float32'))
        out = root/'results'
        run('--input', data, '--output', out, '--plots', 1, '--copy')
        r = pd.read_csv(out/'results.csv').set_index('file')
        truth = r.index.str.startswith('impact')
        assert len(r) == 154
        assert (r.loc[truth, 'label'] == '不正常').all()
        assert (r.loc[~truth, 'label'] == '正常').all()
        assert len(list((out/'plots').glob('*.png'))) == 1
        assert len(list((out/'classified').rglob('*.wav'))) == 154
        ev = pd.read_csv(out/'events.csv')
        locations = ev[ev.file == 'impact_120.wav']
        for t in [.5, 1, 1.5]:
            assert ((locations.start_s <= t) & (locations.end_s >= t)).any()
        run('--input', data, '--output', root/'reuse', '--model', out/'model.json', '--plots', 0)
        second = pd.read_csv(root/'reuse'/'results.csv').set_index('file')
        np.testing.assert_allclose(r.score.sort_index(), second.score.sort_index())
        model = json.loads((out/'model.json').read_text())
        cfg = model['config']
        # Pearson moment definition, with no implicit excess-kurtosis conversion.
        row, f = extract(data/'normal_000.wav', cfg)
        _, x = wavfile.read(data/'normal_000.wav'); x=x.astype(float);x-=x.mean()
        n = round(sr*cfg['frame_ms']/1000)
        np.testing.assert_allclose(f['kurtosis'].iloc[0], kurtosis(x[:n], fisher=False, bias=True))
        # All-normal input is allowed to remain a single normal class.
        normals = r.loc[~truth].reset_index()
        normal_model = fit_model(normals, cfg)
        assert all(not decision(scores([v[k] for k in normal_model['features']], normal_model),cfg)[0]
                   for v in normals.to_dict('records'))
        edges = root/'edges';edges.mkdir()
        wavfile.write(edges/'silence.wav', sr, np.zeros(sr, np.int16))
        wavfile.write(edges/'short.wav', sr, np.array([1,2,3], np.int16))
        (edges/'broken.wav').write_bytes(b'not a WAV')
        wavfile.write(edges/'stereo.wav', sr, np.column_stack([x,-x]).astype('float32'))
        run('--input', edges, '--output', root/'edge_result', '--model', out/'model.json', '--plots', 0)
        errors = pd.read_csv(root/'edge_result'/'errors.csv')
        assert set(errors.file) == {'silence.wav','short.wav','broken.wav'}
        stereo, _ = extract(edges/'stereo.wav', cfg)
        np.testing.assert_allclose(stereo['cf_max'], row['cf_max'], rtol=1e-5)
        # Check known-normal baseline mode independently of batch fitting.
        ref=root/'normal_reference';ref.mkdir()
        import shutil
        for p in sorted(data.glob('normal*.wav'))[:25]:
            shutil.copy2(p, ref/p.name)
        run('--input', edges, '--normal-dir', ref, '--output', root/'normal_result', '--plots', 0)
        assert json.loads((root/'normal_result'/'model.json').read_text())['baseline_source']=='confirmed_normal'
        print('PASS: 154 synthetic files (120 noise / 34 impact), event locations, plots, copies,')
        print('model reuse, all-normal batch, Pearson kurtosis, stereo and quality failures, normal baseline.')
        print('Synthetic results do not establish accuracy on real sensor recordings.')


if __name__ == '__main__':
    main()
