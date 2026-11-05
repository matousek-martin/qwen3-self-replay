import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ops import replay_notification_image as report
from experiments.replay_scaled.notify import deliver_once

spec=importlib.util.spec_from_file_location('telegram_helper','/srv/playground/telegram_notify.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

class NotificationTests(unittest.TestCase):
    def test_official_uses_geometric_savings(self):
        base={d:{'accuracy':.5,'tokens':100.,'thinking':80.} for d,_ in report.DOMAINS}
        metrics={d:{'accuracy':.6,'tokens':50.,'thinking':40.} for d,_ in report.DOMAINS}
        metrics['coding']['tokens']=100.
        result=report.add_savings(metrics,base,True)
        self.assertAlmostEqual(result['overall']['saving'],100*(1-(.5*.5*.5*1.)**.25))
        self.assertAlmostEqual(result['overall']['thinking_saving'],50.)
        self.assertIsNone(report.saving(0,1))
        self.assertIsNone(report.saving(None,1))

    def test_pending_full_not_misrepresented(self):
        r=report.official_report(1280,True)
        self.assertIn('NÁHLED',r['caption'])
        self.assertFalse(r['approx'])
        if r['points'][-1]['metrics'] is None:
            with self.assertRaises(ValueError):report.official_report(1280)
        self.assertTrue(all(p['step']<=1280 for p in r['points']))

    def test_completed_official_matches_authoritative_comparison(self):
        r=report.official_report(1024)
        self.assertAlmostEqual(r['points'][-1]['metrics']['overall']['saving'],43.89189267302801,places=1)
        with tempfile.TemporaryDirectory() as tmp:
            path=report.render(r,Path(tmp)/'image.png')
            self.assertGreater(path.stat().st_size,10000)

    def test_uncertain_photo_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def fail(message):calls.append(message);raise TimeoutError()
            with self.assertRaises(TimeoutError):deliver_once(Path(tmp)/'n.db','result','image',fail)
            self.assertFalse(deliver_once(Path(tmp)/'n.db','result','image',calls.append))
            self.assertEqual(len(calls),1)

    def test_photo_transport_and_safe_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'photo.png';path.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
            config={'token':'123:FAKE_TEST_TOKEN','chat_id':1}
            with patch.object(helper.urllib.request,'urlopen',return_value=io.BytesIO(b'{"ok":true}')) as call:
                helper.send_photo(config,path,'ThinkingCap test')
                req=call.call_args.args[0]
                self.assertTrue(req.full_url.endswith('/sendPhoto'))
                self.assertIn(b'filename="report.png"',req.data)
            with patch.object(helper.urllib.request,'urlopen',side_effect=TimeoutError()):
                with self.assertRaisesRegex(RuntimeError,'uncertain') as raised:helper.send_photo(config,path,'test')
                self.assertNotIn(config['token'],str(raised.exception))

    def test_renderer_errors_never_send_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json';config.write_text(json.dumps({'token':'123:FAKE_TEST_TOKEN','chat_id':1}))
            with patch.object(helper,'CONFIG',config),patch.object(helper,'render_message',side_effect=RuntimeError('render failed')),patch.object(helper,'api') as network:
                with self.assertRaises(RuntimeError):helper.send('test')
                network.assert_not_called()

if __name__=='__main__':unittest.main()
