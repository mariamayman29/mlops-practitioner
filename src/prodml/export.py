import pickle
import time
import numpy as np
import logging
import onnxruntime as rt
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from .logging_conf import setup_logging
from .config import Config
from .data import load_data, split_data
from .features import process_data

logger = logging.getLogger(__name__)


def export_model():
    df = process_data(load_data())
    _, df_val = split_data(df)
    df_val_500 = df_val.head(500)

    with open(Config.model_path, "rb") as f:
        dv, lr = pickle.load(f)

    val_dicts = df_val_500[["PU_DO", "trip_distance"]].to_dict(orient="records")
    X_val_500 = dv.transform(val_dicts).toarray().astype(np.float32)

    initial_type = [("float_input", FloatTensorType([None, X_val_500.shape[1]]))]
    onnx_model = to_onnx(lr, initial_types=initial_type)

    onnx_path = Config.model_path.replace(".pkl", ".onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info(f"ONNX model saved to {onnx_path}")

    times_pkl = []
    for _ in range(10):
        start = time.time()
        pred_pkl = lr.predict(X_val_500)
        times_pkl.append(time.time() - start)

    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    times_onnx = []
    for _ in range(10):
        start = time.time()
        pred_onnx = sess.run(None, {input_name: X_val_500})[0]
        times_onnx.append(time.time() - start)

    pred_onnx = pred_onnx.flatten()
    assert np.allclose(pred_pkl, pred_onnx, atol=1e-4), "Parity test failed!"
    logger.info("Parity test passed")

    logger.info(
        f"Pickle - Mean: {np.mean(times_pkl)*1000:.2f}ms | p95: {np.percentile(times_pkl, 95)*1000:.2f}ms"
    )
    logger.info(
        f"ONNX   - Mean: {np.mean(times_onnx)*1000:.2f}ms | p95: {np.percentile(times_onnx, 95)*1000:.2f}ms"
    )


if __name__ == "__main__":
    setup_logging()
    export_model()
