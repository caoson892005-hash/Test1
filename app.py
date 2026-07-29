"""Webcam person detection and local social-region analysis with LLaVA."""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import ollama
from ultralytics import YOLO


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phát hiện người bằng webcam và phân tích vùng xã hội bằng LLaVA."
    )
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số camera.")
    parser.add_argument(
        "--detector",
        default="yolo11n.pt",
        help="Model Ultralytics; bản nano phù hợp cho xử lý thời gian thực.",
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument(
        "--social-distance",
        type=float,
        default=1.8,
        help="Khoảng cách tâm chuẩn hóa tối đa để tạo cặp xã hội.",
    )
    parser.add_argument(
        "--vlm-interval",
        type=float,
        default=8.0,
        help="Số giây tối thiểu giữa hai lần gọi VLM.",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Chỉ detect người, không gửi ảnh sang LLaVA.",
    )
    return parser.parse_args()


def detect_people(model: YOLO, frame: np.ndarray, confidence: float) -> list[Box]:
    result = model.predict(
        source=frame,
        classes=[0],  # COCO class 0 = person
        conf=confidence,
        verbose=False,
    )[0]
    people: list[Box] = []
    if result.boxes is None:
        return people

    for xyxy, score in zip(
        result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
    ):
        x1, y1, x2, y2 = (int(value) for value in xyxy)
        people.append(Box(x1, y1, x2, y2, float(score)))
    return people


def normalized_distance(first: Box, second: Box) -> float:
    first_center = np.asarray(first.center)
    second_center = np.asarray(second.center)
    average_height = max((first.height + second.height) / 2, 1)
    return float(np.linalg.norm(first_center - second_center) / average_height)


def select_social_pair(
    people: list[Box], distance_threshold: float
) -> tuple[Box, Box] | None:
    """Return the closest plausible pair; proximity alone is not proof of talking."""
    best_pair: tuple[Box, Box] | None = None
    best_distance = float("inf")
    for index, first in enumerate(people):
        for second in people[index + 1 :]:
            distance = normalized_distance(first, second)
            if distance < distance_threshold and distance < best_distance:
                best_pair = (first, second)
                best_distance = distance
    return best_pair


def social_crop(frame: np.ndarray, pair: tuple[Box, Box]) -> np.ndarray:
    height, width = frame.shape[:2]
    first, second = pair
    x1 = min(first.x1, second.x1)
    y1 = min(first.y1, second.y1)
    x2 = max(first.x2, second.x2)
    y2 = max(first.y2, second.y2)
    margin_x = int((x2 - x1) * 0.15)
    margin_y = int((y2 - y1) * 0.15)
    return frame[
        max(0, y1 - margin_y) : min(height, y2 + margin_y),
        max(0, x1 - margin_x) : min(width, x2 + margin_x),
    ].copy()


def encode_jpeg(image: np.ndarray) -> bytes:
    max_side = 768
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1:
        image = cv2.resize(
            image, (int(width * scale), int(height * scale)), cv2.INTER_AREA
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("Không thể mã hóa ảnh vùng xã hội.")
    return encoded.tobytes()


def ask_vlm(image: np.ndarray, model_name: str) -> str:
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": (
                    "Ảnh này là vùng chứa hai người ở gần nhau. "
                    "Hãy đánh giá thận trọng liệu họ có đang tương tác trực tiếp "
                    "hoặc nói chuyện với nhau không. Trả lời tiếng Việt trong tối "
                    "đa 2 câu và nói rõ nếu chỉ từ một ảnh thì chưa đủ chắc chắn."
                ),
                "images": [encode_jpeg(image)],
            }
        ],
        options={"temperature": 0, "num_predict": 160},
    )
    return response.message.content.strip()


def draw_people(frame: np.ndarray, people: list[Box]) -> None:
    for index, person in enumerate(people, start=1):
        cv2.rectangle(
            frame, (person.x1, person.y1), (person.x2, person.y2), (0, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"Person {index}: {person.confidence:.2f}",
            (person.x1, max(20, person.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )


def draw_social_region(frame: np.ndarray, pair: tuple[Box, Box]) -> None:
    first, second = pair
    x1, y1 = min(first.x1, second.x1), min(first.y1, second.y1)
    x2, y2 = max(first.x2, second.x2), max(first.y2, second.y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
    cv2.putText(
        frame,
        "Candidate social region",
        (x1, min(frame.shape[0] - 10, y2 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 165, 255),
        2,
    )


def main() -> None:
    args = parse_args()
    model_name = os.getenv("OLLAMA_MODEL", "llava:7b")
    vlm_enabled = not args.no_vlm

    detector = YOLO(args.detector)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Không mở được camera {args.camera}. Hãy thử --camera 1."
        )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")
    pending: Future[str] | None = None
    last_vlm_call = 0.0
    last_vlm_text = "VLM chưa được gọi"
    person_was_present = False
    social_pair_was_present = False
    result_lock = threading.Lock()

    print("Đang chạy. Nhấn q để thoát.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Không đọc được frame từ camera.")
                break

            people = detect_people(detector, frame, args.confidence)
            person_is_present = bool(people)
            if person_is_present and not person_was_present:
                print(f"Đã tìm thấy người! Số người trong khung hình: {len(people)}")
            elif person_is_present and len(people) > 1:
                print(
                    f"\rĐã tìm thấy người! Số người trong khung hình: {len(people)}",
                    end="",
                    flush=True,
                )
            person_was_present = person_is_present

            pair = select_social_pair(people, args.social_distance)
            if pair is not None:
                draw_social_region(frame, pair)
                now = time.monotonic()
                if not social_pair_was_present:
                    if vlm_enabled:
                        print(
                            "\nĐã bật VLM: phát hiện 2 người ở gần nhau "
                            "và đang tương tác với nhau."
                        )
                    else:
                        print(
                            "\nPhát hiện 2 người ở gần nhau và đang tương tác, "
                            "nhưng VLM đang bị tắt bởi --no-vlm."
                        )
                if (
                    vlm_enabled
                    and pending is None
                    and now - last_vlm_call >= args.vlm_interval
                ):
                    pending = executor.submit(
                        ask_vlm, social_crop(frame, pair), model_name
                    )
                    last_vlm_call = now
                    print("\nĐã gửi vùng xã hội sang VLM...")
            social_pair_was_present = pair is not None

            if pending is not None and pending.done():
                try:
                    with result_lock:
                        last_vlm_text = pending.result()
                    print(f"\nKết quả VLM: {last_vlm_text}")
                except Exception as error:  # Keep camera alive if Ollama fails.
                    last_vlm_text = f"Lỗi VLM: {error}"
                    print(f"\n{last_vlm_text}")
                pending = None

            draw_people(frame, people)
            if pair is not None and vlm_enabled:
                status = "VLM ON | 2 people are interacting"
            elif pair is not None:
                status = "VLM OFF | 2 people are interacting"
            else:
                status = f"People: {len(people)} | Waiting for 2 nearby people"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (20, 20, 20), -1)
            cv2.putText(
                frame,
                status,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Social VLM Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
