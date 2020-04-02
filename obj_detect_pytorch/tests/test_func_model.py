import os
import unittest
import obj_detect_pytorch.config as cfg
import obj_detect_pytorch.models.deepaas_api as deepaas_api
from PIL import Image

class TestModelFunc(unittest.TestCase):
    
    def setUp(self):
        fpath = os.path.join(cfg.BASE_DIR, 'obj_detect_pytorch',
                             'tests', 'inputs', 'bear_test.jpg')
        file = Image.open(fpath)
        self.prob = 0.8
        args = {'files': file, 'outputpath': "temp", 'outputtype' : "json", 
                'threshold' : self.prob, 'box_thickness' : 2,
                'text_size' : 1, 'text_thickness' : 2,
                'model_name' : "COCO", 'accept' : "application/json"}
        self.pred = deepaas_api.predict(**args)
        
     
    def test_model_prediction(self):
        """
        Test that predict() returns dict.
        """
        self.assertTrue(type(self.pred) is dict)
        
    def test_predict(self):
        """
        Test that predict() returns right prediction values.
        """
        prob = self.pred.get('results')[0].get('probability')
        
        print("prob bear: ", prob)
        assert float(prob) >= self.prob

if __name__ == '__main__':
    unittest.main()

