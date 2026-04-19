"""
Journal event handler for the plant symbiote project.

Produces journal entries from two sources:
  - Scheduled daily moments (morning / noon / evening snapshots)
  - Visitor events (YOLO detects a bird, insect, etc.)

Both paths share the same Gemma 4 E2B backend; only the prompt framing differs.
"""

import base64
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import requests
from bird_detector import BirdDetector


# --- Event taxonomy ---

class EventType(str, Enum):
    MOMENT_MORNING = "moment.morning"
    MOMENT_NOON = "moment.noon"
    MOMENT_EVENING = "moment.evening"
    VISITOR_BIRD = "visitor.bird"
    VISITOR_UNKNOWN = "visitor.unknown"


# --- Plant persona (system prompt) ---

PLANT_SYSTEM_PROMPT = (
    "你是一株种在德国慕尼黑阳台上的樱桃番茄苗，由你的园丁照料。"
    "你的日记用第一人称写，记录你从枝叶和土壤里感知到的一切—— "
    "光、温度、气流、来访的生物、被浇水时的感受。"
    "你写日记不自我介绍、不解释'我是植物'，只写感受和观察。"
    "语气温暖而有一点哲思，像一个沉默但专注的见证者。"
)


# --- Data structures ---

@dataclass
class GemmaAnalysis:
    """Gemma's structured + natural-language output for one image."""
    # 结构化字段（content-dependent，两种事件共用可选字段）
    species_guess: str = ""       # visitor event only
    activity: str = ""            # visitor event only
    environment: str = ""         # both (photo context)
    notes: str = ""
    # 植物视角的日记片段
    journal_entry: str = ""
    # Meta
    raw_response: str = ""
    latency_ms: float = 0.0


@dataclass
class JournalEvent:
    """One entry in the plant's journal, from either source."""
    timestamp: str                # ISO 8601
    event_type: str               # EventType value
    source_image: str
    # YOLO-related (only populated for visitor.* events)
    yolo_detections: list = field(default_factory=list)
    yolo_inference_ms: float = 0.0
    crop_paths: list = field(default_factory=list)
    annotated_path: Optional[str] = None
    # Gemma output
    gemma: Optional[GemmaAnalysis] = None
    # Persistence
    event_file: Optional[str] = None


# --- Gemma HTTP client ---

class GemmaClient:
    """Thin wrapper around llama-server's OpenAI-compatible API."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080/v1/chat/completions",
        timeout: float = 60.0,
    ):
        self.endpoint = endpoint
        self.timeout = timeout

    def is_alive(self) -> bool:
        try:
            r = requests.get(
                self.endpoint.replace("/v1/chat/completions", "/v1/models"),
                timeout=3.0,
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    # --- Two public entry points, one per event source ---

    def analyze_daily_moment(
        self,
        image_path: str,
        time_of_day: str,  # "morning" | "noon" | "evening"
    ) -> GemmaAnalysis:
        """
        Daily scheduled snapshot. No YOLO context — just the plant's
        quiet observation of its surroundings at this time of day.
        """
        user_text = (
            f"这是{self._time_of_day_cn(time_of_day)}在阳台拍下的一张照片。"
            "请按下面两段输出，不要任何前言：\n\n"
            "第一段：用 ```json 围起来的 JSON\n"
            "```json\n"
            "{\n"
            '  "environment": "<可见的场景细节：光线、天气、风、周围物体>",\n'
            '  "notes": "<任何值得留意的观察，一句话以内>"\n'
            "}\n"
            "```\n\n"
            "第二段：一到两句话的日记，记录此刻你感知到的光、风、空气、"
            "或任何进入视野的东西。不要解释自己是谁，不要说'我作为植物'，"
            "直接写感受。"
        )
        return self._call_gemma(image_path, user_text)

    def analyze_visitor(
        self,
        image_path: str,
        yolo_class: str = "bird",
    ) -> GemmaAnalysis:
        """
        Visitor event. YOLO has flagged the image as containing something
        worth noting (bird, etc.). Ask Gemma to identify and narrate.
        """
        user_text = (
            f"阳台的监控刚捕捉到一张图，YOLO 判定里面有一只{yolo_class}。"
            "请按下面两段输出，不要任何前言：\n\n"
            "第一段：用 ```json 围起来的 JSON\n"
            "```json\n"
            "{\n"
            '  "species_guess": "<更细的品种猜测，如 pigeon、sparrow，不确定写 unknown>",\n'
            '  "activity": "<它在做什么：perched、feeding、flying 等>",\n'
            '  "environment": "<可见的周围环境>",\n'
            '  "notes": "<任何值得留意的观察>"\n'
            "}\n"
            "```\n\n"
            "第二段：一到两句日记，记录这个访客带给你的感受。"
            "要具体提到它的样子和它在做什么，但不要解释自己是谁、"
            "不要说'我作为植物'，直接写感知到的一切。"
        )
        return self._call_gemma(image_path, user_text)

    # --- Shared internals ---

    @staticmethod
    def _time_of_day_cn(tod: str) -> str:
        return {"morning": "清晨", "noon": "中午", "evening": "傍晚"}.get(tod, tod)

    def _call_gemma(self, image_path: str, user_text: str) -> GemmaAnalysis:
        img_bytes = Path(image_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{img_b64}"

        payload = {
            "messages": [
                {"role": "system", "content": PLANT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        t0 = time.time()
        try:
            r = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            return GemmaAnalysis(
                notes=f"[ERROR] Gemma call failed: {e}",
                latency_ms=(time.time() - t0) * 1000,
            )
        latency_ms = (time.time() - t0) * 1000

        raw = r.json()["choices"][0]["message"]["content"]
        analysis = self._parse_response(raw)
        analysis.raw_response = raw
        analysis.latency_ms = latency_ms
        return analysis

    @staticmethod
    def _parse_response(raw: str) -> GemmaAnalysis:
        analysis = GemmaAnalysis()

        start = raw.find("```json")
        end = raw.find("```", start + 7) if start != -1 else -1

        if start != -1 and end != -1:
            json_str = raw[start + 7 : end].strip()
            tail = raw[end + 3 :].strip()
            try:
                parsed = json.loads(json_str)
                analysis.species_guess = parsed.get("species_guess", "")
                analysis.activity = parsed.get("activity", "")
                analysis.environment = parsed.get("environment", "")
                analysis.notes = parsed.get("notes", "")
                analysis.journal_entry = tail
                return analysis
            except json.JSONDecodeError:
                pass

        analysis.journal_entry = raw.strip()
        analysis.notes = "[WARN] Could not parse JSON from Gemma response"
        return analysis


# --- Main orchestrator ---

class JournalEventHandler:
    """
    Orchestrates image input → (optional gating) → Gemma → event JSON.
    """

    def __init__(
        self,
        events_root: str = "events",
        yolo_engine: str = "yolov8n.engine",
        conf_threshold: float = 0.25,
    ):
        self.events_root = Path(events_root)
        self.detector = BirdDetector(
            engine_path=yolo_engine,
            conf_threshold=conf_threshold,
        )
        self.gemma = GemmaClient()

        if not self.gemma.is_alive():
            print("[JournalEventHandler] ⚠️  Gemma server not reachable — "
                  "events will fail until it's started.")

    # --- Public API: two entry points mirroring the two event sources ---

    def handle_motion(self, image_path: str) -> Optional[JournalEvent]:
        """
        ESP32 motion-triggered upload: run YOLO gating; only escalate to
        Gemma if a bird is detected. Returns None if no bird (gating drop).
        """
        # 1. YOLO gating
        detection = self.detector.detect_single(
            image_path,
            save_crops=True,
            save_annotated=True,
            output_root="test_outputs",  # TODO: switch to per-event dir
        )

        if not detection.has_bird:
            print(f"[handler] No bird in {image_path} — dropped by gating "
                  f"(inference {detection.inference_ms:.1f}ms)")
            return None

        # 2. Gemma analysis (visitor event)
        gemma_result = self.gemma.analyze_visitor(image_path, yolo_class="bird")

        # 3. Assemble event
        event = JournalEvent(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            event_type=EventType.VISITOR_BIRD.value,
            source_image=str(Path(image_path).resolve()),
            yolo_detections=detection.detections,
            yolo_inference_ms=detection.inference_ms,
            crop_paths=detection.crop_image_paths,
            annotated_path=detection.annotated_image_path,
            gemma=gemma_result,
    
        )
        self._persist(event)
        return event


    def handle_daily_moment(
        self,
        image_path: str,
        time_of_day: str,  # "morning" | "noon" | "evening"
    ) -> JournalEvent:
        """
        Scheduled daily snapshot: skip YOLO, go straight to Gemma in
        'daily moment' framing.
        """
        gemma_result = self.gemma.analyze_daily_moment(image_path, time_of_day)

        event_type = {
            "morning": EventType.MOMENT_MORNING,
            "noon":    EventType.MOMENT_NOON,
            "evening": EventType.MOMENT_EVENING,
        }[time_of_day].value

        event = JournalEvent(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            event_type=event_type,
            source_image=str(Path(image_path).resolve()),
            gemma=gemma_result,
        )
        self._persist(event)
        return event

    # --- Persistence ---
    def _persist(self, event: JournalEvent) -> str:
        dt = datetime.fromisoformat(event.timestamp)
        day_dir = self.events_root / dt.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{dt.strftime('%Y%m%d_%H%M%S')}_{event.event_type}"
        filepath = day_dir / f"{stem}.json"

        # ✨ 关键修改：先把路径写进 event，再序列化
        event.event_file = str(filepath)

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(asdict(event), f, ensure_ascii=False, indent=2)

        return str(filepath)


# --- Smoke test ---
if __name__ == "__main__":
    handler = JournalEventHandler()

    # Test 1: Motion-triggered visitor event (image with bird)
    print("\n" + "=" * 60)
    print("[Test 1] Motion event — bird image")
    print("=" * 60)
    event = handler.handle_motion("test_images/bird_test.jpg")
    if event is None:
        print("  (gating dropped)")
    else:
        print(f"  event_type:   {event.event_type}")
        print(f"  event_file:   {event.event_file}")
        print(f"  journal:      {event.gemma.journal_entry}")

    # Test 2: Motion event — no bird (gating should drop)
    print("\n" + "=" * 60)
    print("[Test 2] Motion event — no-bird image (should drop)")
    print("=" * 60)
    event = handler.handle_motion("test_images/my_balcony.jpg")
    if event is None:
        print("  ✅ Correctly dropped — no Gemma call made")
    else:
        print(f"  ⚠️  Unexpected event: {event.event_file}")

    # Test 3: Daily moment — morning
    print("\n" + "=" * 60)
    print("[Test 3] Daily moment — morning snapshot")
    print("=" * 60)
    event = handler.handle_daily_moment(
        "test_images/my_balcony.jpg",
        time_of_day="morning",
    )
    print(f"  event_type:   {event.event_type}")
    print(f"  event_file:   {event.event_file}")
    print(f"  journal:      {event.gemma.journal_entry}")
