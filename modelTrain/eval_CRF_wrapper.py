import joblib
from pathlib import Path
from evalCRF import sentence_features, bmes_to_segmented

class CRFSegmenter:
    def __init__(self, model_path="crfModel/crf_poem_seg.pkl"):
        
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"CRF 模型文件不存在：{path}")
        self.crf_model = joblib.load(path)

    def segment(self, text):
        if not text:
            return ""
        labels = self.crf_model.predict_single(sentence_features(text))
        return bmes_to_segmented(text, labels)
