
from  models.infer import Inference
import argparse
import time
import cv2

## TODO  : Create a Inference Model fo 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="/dev/video5")
    parser.add_argument("--imgsz", type=str, default="224")
    parser.add_argument("--confidence_thres", type=float, default=0.6)
    parser.add_argument("--iou_thres", type=float, default=0.5)
    parser.add_argument("--inference_engine", type=str, default="openvino")
    parser.add_argument("--inference_version", type=str, default="0.0.1")
    parser.add_argument("--model_version", type=str, default="v11")
    return parser


def main():

    args = parse_args()


    args.source = "data/test_videos/rp.avi"


    model = Inference(kwargs= vars(args))

    cap = cv2.VideoCapture(args.source)

    while cap.isOpened():
        try:
            ret, frame = cap.read()
        # Crop from the middle 

            frame = frame[200 -112 :200 + 112, 224 - 112:224 + 112]
            print(frame.shape)

            if not ret: 
                break
            
            start_time = time.time()
            results , loc = model(frame)
            end_time = time.time()
            cv2.imshow("frame", results)
            print(f"Inference time: {end_time - start_time:.2f} seconds")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            print(e)
            break



if __name__ == "__main__":
    main()