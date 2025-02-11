from ultralytics import YOLOWorld
import numpy as np
import dtlpy as dl
import onnxruntime
import tempfile
import logging
import base64
import torch
import json
import time
import clip
import cv2
import os
from PIL import Image

logger = logging.getLogger("YOLO-SAM-TOOLBAR")

# set max image size
Image.MAX_IMAGE_PIXELS = 933120000


class Runner(dl.BaseServiceRunner):
    def __init__(self, dl):
        if os.path.isfile("/tmp/app/weights/yolov8l-worldv2.pt"):
            path = "/tmp/app/weights/yolov8l-worldv2.pt"
        else:
            path = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-worldv2.pt"
        self.yolo_model = YOLOWorld(path)
        # warmup
        self.yolo_model.predict("assets/000000001296.jpg")
        self.device = "cpu"
        self.batch = 80
        self.clip_model, _ = clip.load("ViT-B/32")

        # find SAM global service
        p = dl.projects.get("DataloopTasks")
        self.sam_service = p.services.get(service_name="global-sam")

        # load sam onnx
        if os.path.isfile("/tmp/app/weights/sam2_hiera_small.decoder.onnx"):
            onnx_model_path = "/tmp/app/weights/sam2_hiera_small.decoder.onnx"
        else:
            onnx_model_path = "weights/sam2_hiera_small.decoder.onnx"
        self.sam_ort_session = onnxruntime.InferenceSession(
            onnx_model_path, providers=["CPUExecutionProvider"]
        )
        self.output_name = [a.name for a in self.sam_ort_session.get_outputs()]

    def run_and_upload(self, dl, item: dl.Item):
        annotations = self.run(dl, item)
        item.annotations.upload(annotations)

    def run(self, dl, item: dl.Item):
        """
        Run the YOLO-World and SAM decoder on the item
        """
        # get item's image
        if "bot.dataloop.ai" in dl.info()["user_email"]:
            raise ValueError("This function cannot run with a bot user")
        tic = time.time()
        with tempfile.TemporaryDirectory() as tempf:
            source = item.download(local_path=tempf, overwrite=True)
            # get labels from dataset
            labels = list(item.dataset.labels_flat_dict.keys())
            if len(labels) == 0:
                # if recipe labels are empty, use yolo labels
                labels = list(self.yolo_model.model.names.values())
            yolo_results = self.run_yolo(source, labels)
            collection = self.run_sam(
                dl=dl, item=item, yolo_results=yolo_results, labels=labels
            )
        logger.info(f"Full run took: {time.time() - tic:.2f} seconds")
        return collection.to_json()["annotations"]

    def run_sam(self, dl, item: dl.Item, yolo_results, labels):
        """
        Run the SAM decoder on the item
        """
        tic = time.time()
        ex = dl.services.execute(
            service_id=self.sam_service.id,
            function_name="get_sam_features",
            item_id=item.id,
            project_id=item.project.id,
        )
        ex = dl.executions.wait(execution=ex, timeout=60)
        if ex.latest_status["status"] not in ["success"]:
            raise ValueError(f"Execution failed. ex id: {ex.id}")
        logger.info(f"SAM execution took: {time.time() - tic:.2f} seconds")
        item_bytes = dl.items.get(item_id=ex.output).download(save_locally=False)
        image_embedding_dict = json.load(item_bytes)
        # base64.b64encode(high_res_feats_1).decode('utf-8')
        image_embed = np.frombuffer(
            base64.b64decode(image_embedding_dict["image_embed"]), dtype=np.float32
        ).reshape([1, 256, 64, 64])
        high_res_feats_0 = np.frombuffer(
            base64.b64decode(image_embedding_dict["high_res_feats_0"]), dtype=np.float32
        ).reshape([1, 32, 256, 256])
        high_res_feats_1 = np.frombuffer(
            base64.b64decode(image_embedding_dict["high_res_feats_1"]), dtype=np.float32
        ).reshape([1, 64, 128, 128])
        embed_size = 64
        height = item.height
        width = item.width
        collection = dl.AnnotationCollection()
        for i, d in enumerate(reversed(yolo_results.boxes)):
            c, d_conf, obj_id = int(d.cls), float(d.conf), d.id
            name = labels[c]
            box = d.xyxy.squeeze()
            print(f"{name} {d_conf:.2f} {box}")
            feeds = {
                "image_embed": image_embed,
                "high_res_feats_0": high_res_feats_0,
                "high_res_feats_1": high_res_feats_1,
                "point_coords": np.array(
                    [
                        [
                            [box[0] / width * 1024, box[1] / height * 1024],
                            [box[2] / width * 1024, box[3] / height * 1024],
                        ]
                    ],
                    dtype=np.float32,
                ),
                "point_labels": np.array([[2, 3]], dtype=np.float32),
                "mask_input": torch.randn(
                    1, 1, 4 * embed_size, 4 * embed_size, dtype=torch.float
                )
                .cpu()
                .numpy(),
                "has_mask_input": np.array([0], dtype=np.float32),
            }

            result = self.sam_ort_session.run([self.output_name[0]], feeds)
            mask = cv2.resize(result[0][0][0], (width, height))
            collection.add(
                annotation_definition=dl.Box(
                    label=name, left=box[0], top=box[1], right=box[2], bottom=box[3]
                ),
                object_id=obj_id,
                model_info={"name": "yoloworld", "confidence": d_conf},
            )
            collection.add(
                annotation_definition=dl.Segmentation(geo=mask > 0, label=name),
                object_id=obj_id,
                model_info={"name": "yoloworld", "confidence": d_conf},
            )
        logger.info(f"Total SAM predictions took: {time.time() - tic:.2f} seconds")
        return collection

    def run_yolo(self, source: str, labels: list):
        """
        Run the YOLO-World on the item
        """
        tic = time.time()
        text_token = clip.tokenize(labels).to(self.device)
        txt_feats = [
            self.clip_model.encode_text(token).detach()
            for token in text_token.split(self.batch)
        ]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, len(labels), txt_feats.shape[-1])

        im0s = [cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB)]
        im = self.yolo_model.predictor.preprocess(im0s)
        outputs = self.yolo_model.predictor.model.model.predict(
            x=im, txt_feats=txt_feats
        )
        results = self.yolo_model.predictor.postprocess(outputs, im, im0s)[0]
        logger.info(f"YOLO took: {time.time() - tic:.2f} seconds")
        return results


def test():
    import dtlpy as dl

    dl.setenv("rc")
    # item = dl.items.get(item_id="67646bb3b976103a2af68ae1")  # prod
    item = dl.items.get(item_id="67646bb3b976103a2af68ae1")  # rc

    runner = Runner(dl=dl)
    annotations = runner.run(dl=dl, item=item)
    item.annotations.upload(annotations)


if __name__ == "__main__":
    test()
