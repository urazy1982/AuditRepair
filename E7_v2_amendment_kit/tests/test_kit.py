import importlib.util,pathlib,tempfile,unittest,json,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def mod(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
collect=mod('collect');validate=mod('validate')
class Contracts(unittest.TestCase):
    def setUp(self):self.classes=['World','Sports','Business','Sci/Tech']
    def test_strict_choice(self):
        self.assertEqual(validate.parse_v2({'status':'received','finish_reason':'stop','response_text':' Business\n'},self.classes),'Business')
        for text,finish in [('Business because','stop'),('Answer: Business','stop'),('Business','length')]:
            self.assertIsNone(validate.parse_v2({'status':'received','finish_reason':finish,'response_text':text},self.classes))
    def test_original_reason(self):
        self.assertIsNone(validate.parse_original({'status':'received','finish_reason':'stop','response_text':'Answer: World\nDetails'},'reason',self.classes))
        self.assertEqual(validate.parse_original({'status':'received','finish_reason':'stop','response_text':'Reason.\nAnswer: World'},'reason',self.classes),'World')
    def test_request_hash_order_independent(self):
        a={'item_id':'5','generation':{'temperature':0,'structured_outputs':{'choice':self.classes}}}
        self.assertEqual(collect.request_hash(a),collect.request_hash(json.loads(json.dumps(a))))
    def test_failed_result_not_a_valid_response(self):
        self.assertFalse(collect.exact_response({'status':'request_failed','response_text':'World','finish_reason':'stop'},self.classes))
    def test_source_uses_no_gold(self):
        s=(ROOT/'prepare.py').read_text()
        self.assertNotIn("['gold']",s);self.assertNotIn("['true_label']",s)
    def test_verifier_gate_before_gold(self):
        s=(ROOT/'validate.py').read_text()
        self.assertLess(s.index("if not valid:raise SystemExit(2)"),s.index("import pandas as pd"))
if __name__=='__main__':unittest.main()
