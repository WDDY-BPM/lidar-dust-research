"""Remote synthetic checks for capture repair, diagnostics and frozen experiments."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_dust
import freeze_baseline
from dust.data import file_sha256, fingerprint, read_manifest, write_json
from dust.diagnostics import duplicate_label_differences, inside_boxes, read_boxes
from dust.grouping import (CAPTURE_POLICY, assert_clean_capture_records, capture_day,
                           repair_capture_records)
from prepare_dust import prepare
from evaluate_dust import main as evaluate_main
from train_dust import train
from test_dust_baseline import TinyNetwork


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def record(self, sequence, day, point=None, label=None, sensor="ls64", frame=1):
        stamp = 1609459200 + day * 86400 if day is not None else 4
        path = f"{sensor}/sequence_{sequence:04d}_{sensor}_{frame:04d}_{stamp}_000.bin"
        return {"path": path, "sequence": f"sequence_{sequence:04d}", "sensor": sensor,
                "point_sha256": point or hashlib.sha256(path.encode()).hexdigest(),
                "label_sha256": label or hashlib.sha256((path + '.label').encode()).hexdigest()}

    def fixture(self, work, n=3):
        root = work / "raw"
        for sensor, start in (("m1", 0), ("ouster", 10)):
            for sequence in range(start, start + n):
                record = self.record(sequence, sequence, sensor=sensor)
                path = root / record["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                np.array([[0, 0, 0, sequence + 1], [1, 1, 1, sequence + 4],
                          [2, 2, 0, sequence + 7], [np.nan, 0, 0, 0]], dtype="<f4").tofile(path)
                np.array([12, 9, 25, 0], dtype=np.uint8).tofile(path.with_suffix('.label'))
        args = argparse.Namespace(data_root=str(root), output=str(work / "prepared"), group_map=None,
                                  seed=42, train_ratio=.7, val_ratio=.15, npoint=4096,
                                  block_size=10., stride=5., coord_scale=10.)
        return root, args

    def test_day_and_duplicate_links_bind_whole_sequences_before_deduplication(self):
        rows = [self.record(0, 0, "same", "same-label"), self.record(1, 1, "same", "same-label"),
                self.record(1, 1, frame=2), self.record(2, 0, sensor="m1"), self.record(3, 3)]
        original = copy.deepcopy(rows)
        kept, report = repair_capture_records(rows)
        self.assertEqual(rows, original)
        self.assertEqual(len(kept), 4)
        self.assertEqual(report['excluded_frames'], 1)
        groups = {r['sequence']: r['group'] for r in kept}
        self.assertEqual(groups['sequence_0000'], groups['sequence_0001'])
        self.assertEqual(groups['sequence_0000'], groups['sequence_0002'])
        self.assertNotEqual(groups['sequence_0000'], groups['sequence_0003'])
        again, _ = repair_capture_records(list(reversed(rows)))
        self.assertEqual(kept, again)

    def test_conflicting_labels_and_relative_clocks_are_quarantined(self):
        rows = [self.record(0, 0, "same", "a"), self.record(1, 1, "same", "b"),
                self.record(0, 0, frame=2), self.record(1, 1, frame=2), self.record(2, None)]
        kept, report = repair_capture_records(rows)
        self.assertEqual(len(kept), 2)
        self.assertEqual(report['conflicting_label_groups'], 1)
        self.assertEqual(report['exclusion_reason_counts']['identical_points_conflicting_labels'], 2)
        self.assertEqual(report['exclusion_reason_counts']['unreliable_clock_without_capture_map'], 1)
        self.assertEqual(kept[0]['group'], kept[1]['group'])
        mapping = {f'sequence_{i:04d}': 'reviewed_capture' for i in range(3)}
        mapped, _ = repair_capture_records(rows, mapping)
        self.assertEqual(len(mapped), 3)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            repair_capture_records(rows, {'sequence_0000': 'incomplete'})
        self.assertIsNone(capture_day('ouster/sequence_0146_ouster_0001_4_000.bin'))

    def test_prepare_excludes_before_statistics_and_loader_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, args = self.fixture(work, n=4)
            first = sorted((root/'m1').glob('*.bin'))[0]
            second = sorted((root/'m1').glob('*.bin'))[1]
            for source, sequence, labels in ((first, 98, None), (second, 99, [9, 9, 25, 0])):
                path = source.with_name(source.name.replace(source.stem.split('_')[1], f'{sequence:04d}', 1))
                path.write_bytes(source.read_bytes())
                path.with_suffix('.label').write_bytes(source.with_suffix('.label').read_bytes()
                                                       if labels is None else np.array(labels,dtype=np.uint8).tobytes())
            relative = root/'m1/sequence_0097_m1_0001_4_000.bin'
            np.array([[0, 0, 0, 1e6], [1, 1, 1, 1e6]],dtype='<f4').tofile(relative)
            np.array([12,9],dtype=np.uint8).tofile(relative.with_suffix('.label'))
            manifest = prepare(args)
            self.assertEqual(manifest['preparation_protocol'], CAPTURE_POLICY)
            self.assertEqual(len(manifest['frames']), 7)
            self.assertEqual(len({r['point_sha256'] for r in manifest['frames']}), 7)
            self.assertEqual(sum(r['sequence']=='sequence_0097' for r in manifest['frames']), 0)
            expected=[]
            for record in manifest['frames']:
                if record['split']=='train' and record['sensor']=='m1':
                    points=np.fromfile(root/record['path'],dtype='<f4').reshape(-1,4)
                    expected.extend(np.log1p(points[np.isfinite(points).all(1),3]).tolist())
            self.assertAlmostEqual(manifest['intensity']['m1']['mean'],float(np.mean(expected)),places=6)
            broken=copy.deepcopy(manifest)
            broken['frames'][1]['point_sha256']=broken['frames'][0]['point_sha256']
            broken.pop('fingerprint')
            broken['fingerprint']=fingerprint(broken)
            write_json(work/'broken.json',broken)
            with self.assertRaisesRegex(ValueError,'Duplicate point cloud'):
                read_manifest(work/'broken.json')
            records=copy.deepcopy(manifest['frames'])
            rogue=dict(records[0],path=records[0]['path'].replace('_0001_','_9999_'),
                       group='rogue',point_sha256='different')
            with self.assertRaisesRegex(ValueError,'multiple capture groups'):
                assert_clean_capture_records(records+[rogue])

    def test_duplicate_disagreement_separates_keep_dust_from_ignore_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            records=[]
            for sequence,labels in ((0,[12,9,25,0]),(1,[9,9,1,0])):
                record=self.record(sequence,sequence)
                path=root/record['path']; path.parent.mkdir(parents=True,exist_ok=True)
                np.array([[0,0,0,1],[1,1,1,2],[2,2,0,3],[np.nan,0,0,0]],dtype='<f4').tofile(path)
                np.array(labels,dtype=np.uint8).tofile(path.with_suffix('.label'))
                record.update(label_path=str(Path(record['path']).with_suffix('.label')),
                              point_sha256=file_sha256(path),label_sha256=file_sha256(path.with_suffix('.label')))
                records.append(record)
            # Use an independent clean neighbor so quarantine does not empty the input.
            _, grouping=repair_capture_records(records+[self.record(2,2)])
            differences=duplicate_label_differences(root,records,grouping['duplicates'])
            self.assertEqual(differences[0]['binary_disagreements'],2)
            self.assertEqual(differences[0]['keep_dust_contradictions'],1)
            self.assertEqual(differences[0]['ignore_supervised_differences'],1)

    def test_box_hypotheses_are_explicit_and_do_not_change_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'boxes.txt'
            path.write_text('Car 2 2 4 0 0 0 0\n')
            boxes=read_boxes(path)
            points=np.array([[0,0,0],[3,0,0],[0,0,1.5]],dtype=np.float32)
            self.assertEqual(inside_boxes(points,boxes,'Car','center').tolist(),[True,False,False])
            self.assertEqual(inside_boxes(points,boxes,'Car','bottom').tolist(),[True,False,True])

    def test_saved_prediction_is_authoritative_at_float32_threshold_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            root,args=self.fixture(work)
            manifest=prepare(args)
            record=manifest['frames'][0]
            visual=work/'saved'; prefix=visual/Path(record['path']).with_suffix('')
            prefix.parent.mkdir(parents=True,exist_ok=True)
            np.savez(str(prefix)+'_prediction.npz',semantic_label=np.array([12,9,25,0],dtype=np.uint8),
                     finite_index=np.array([0,1,2]),threshold=np.array(.5),
                     dust_probability=np.array([.49999999,.9,.1,np.nan],dtype=np.float32),
                     prediction=np.array([0,1,0,-100],dtype=np.int16))
            result=diagnose_dust.inspect_case(root,work/'review',record,{'tn':1,'fp':0,'fn':0,'tp':1},
                        manifest,{'threshold':.5},12,existing_visualizations=visual)
            self.assertTrue(result['confusion_matches_saved_frame'])
            self.assertEqual(result['float32_probability_threshold_disagreements'],1)

    def test_diagnostic_forward_and_frozen_protocol_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            root,args=self.fixture(work)
            manifest=prepare(args)
            training=argparse.Namespace(data_root=str(root),manifest=str(work/'prepared/manifest.json'),
                    output=str(work/'run'),resume=None,device='cpu',epochs=1,batch_size=2,workers=0,
                    samples_per_frame=1,seed=42,lr=.001,weight_decay=.0001,threshold=.5,eval_every=10,
                    eval_batch_size=2,max_train_frames=None,max_val_frames=None,no_augment=False)
            evaluation=argparse.Namespace(data_root=str(root),manifest=training.manifest,
                    checkpoint=str(work/'run/best.pt'),output=str(work/'test'),split='test',device='cpu',
                    max_frames=None,visualize_per_sensor=0,allow_smoke=False,backend='eager')
            with patch('train_dust.get_model',TinyNetwork):
                train(training)
            with patch('evaluate_dust.get_model',TinyNetwork):
                evaluate_main(evaluation)
            sequence=next(r['sequence'] for r in manifest['frames'] if r['split']=='test')
            diagnosis=argparse.Namespace(data_root=str(root),manifest=training.manifest,results=evaluation.output,
                    checkpoint=evaluation.checkpoint,sequence=sequence,semantic_id=12,review_frames=2,
                    device='cpu',backend='eager',output=str(work/'diagnosis'))
            with patch('diagnose_dust.get_model',TinyNetwork):
                result=diagnose_dust.main(diagnosis)
            self.assertTrue(result['all_review_confusions_match'])
            self.assertTrue(result['all_review_predictions_available'])
            self.assertEqual(result['input_fingerprint'],manifest['input_fingerprint'])
            freeze_args=argparse.Namespace(data_root=str(root),manifest=training.manifest,
                    diagnostics=str(work/'diagnosis/summary.json'),output=str(work/'protocol with space'))
            protocol=freeze_baseline.freeze(freeze_args)
            checked,_=freeze_baseline.check_protocol(work/'protocol with space/protocol.json')
            self.assertEqual(protocol,checked)
            script=work/'protocol with space/run_baseline.sh'
            subprocess.run(['bash','-n',str(script)],check=True)
            text=script.read_text()
            self.assertIn('--eval-every 10',text)
            self.assertNotIn('\n+      ',text)
            self.assertIn('for seed in 42 43 44',text)
            # Metadata-only mock checkpoint exercises restart guards without training another model.
            config={key:freeze_baseline.SETTINGS[key] for key in ('batch_size','samples_per_frame','lr',
                    'weight_decay','threshold','eval_every','eval_batch_size')}
            config.update(seed=42,manifest_fingerprint=protocol['manifest_fingerprint'],source=protocol['source'],
                    model='pointnet2_dust_msg',augmentation='z_rotation',max_train_frames=None,max_val_frames=None,
                    evaluation_seed=42,is_smoke_run=False,training_order='epoch_permutation_v1',initialization=None)
            state={'format_version':2,'config':config,'training_budget_epochs':100,'next_epoch':14,'next_batch':2,
                   'resume_lineage':[],'eval_backend':'cuda-graph'}
            weights=work/'metadata.pt'; torch.save(state,weights)
            self.assertEqual(freeze_baseline.checkpoint_status(weights,protocol,42),'partial')
            state['next_epoch'],state['next_batch']=100,0; torch.save(state,weights)
            self.assertEqual(freeze_baseline.checkpoint_status(weights,protocol,42),'complete')
            with self.assertRaisesRegex(ValueError,'configuration'):
                freeze_baseline.checkpoint_status(weights,protocol,43)
            report=json.loads((work/'test/metrics.json').read_text())
            report.update(training_config=config,training_budget_epochs=100,checkpoint_source=protocol['source'],
                          checkpoint_sha256=file_sha256(weights),inference_backend='cuda-graph',
                          evaluation_batch_size=4,validation_schedule={'initial_eval_every':10,'changes':[]})
            write_json(work/'frozen_report.json',report)
            freeze_baseline.check_report(work/'frozen_report.json',weights,protocol,manifest,42)
            weights.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError,'match this frozen'):
                freeze_baseline.check_report(work/'frozen_report.json',weights,protocol,manifest,42)


if __name__=='__main__':
    unittest.main()
