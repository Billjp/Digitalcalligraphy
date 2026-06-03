import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import time
import platform
import math

# wholebody12 クラスID定義
CLASS_BODY  = 0
CLASS_HEAD  = 3
CLASS_FACE  = 4
CLASS_HAND  = 8  # hand (検証済み: score=0.954で正常検出)

class WholeBodyEstimator:
    """
    PINTO_model_zoo の YOLOX-WholeBody12 ONNXモデル。
    出力: (N, 7) = [batch_id, class_id, score, x1, y1, x2, y2]
    ※ キーポイント出力は存在しない（バウンディングボックス検出器）
    """
    def __init__(self, model_path="yolox_s_wholebody12_post_0190_1x3x480x960.onnx"):
        self.model_path = model_path
        self.session = None
        self.in_h, self.in_w = 480, 960

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                self.model_path, providers=['CPUExecutionProvider']
            )
            print(f"[OK] ONNX model loaded: {self.model_path}")
        except Exception as e:
            print(f"[Error] {e}")

    def infer(self, frame_bgr):
        """
        frame_bgr: OpenCVのBGR形式のフレーム（flip済み）
        Returns: dict of detected regions, or None
        """
        if self.session is None:
            return None

        h, w = frame_bgr.shape[:2]
        input_name = self.session.get_inputs()[0].name

        # 前処理: BGRのまま resize → CHW → float32
        resized = cv2.resize(frame_bgr, (self.in_w, self.in_h))
        tensor = resized.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        outputs = self.session.run(None, {input_name: tensor})
        boxes = outputs[0]  # shape: (N, 7)

        if len(boxes) == 0:
            return None

        # 画面座標への変換スケール
        sx = w / self.in_w
        sy = h / self.in_h

        def best_box(class_id):
            """指定クラスIDで最高スコアのボックスを返す"""
            candidates = [b for b in boxes if int(b[1]) == class_id]
            if not candidates:
                return None
            best = max(candidates, key=lambda b: b[2])
            _, _, score, x1, y1, x2, y2 = best
            cx = (x1 + x2) / 2 * sx
            cy = (y1 + y2) / 2 * sy
            bw = (x2 - x1) * sx
            bh = (y2 - y1) * sy
            return {"cx": cx, "cy": cy, "w": bw, "h": bh, "score": score}

        res = {
            "hand":  best_box(CLASS_HAND),
            "body":  best_box(CLASS_BODY),
            "head":  best_box(CLASS_HEAD),
            "face":  best_box(CLASS_FACE),
        }

        # 何も検出されなかった場合
        if all(v is None for v in res.values()):
            return None

        return res

    def close(self):
        pass


class DigitalCalligraphyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Calligraphy App - YOLOX WholeBody12")
        self.root.geometry("1280x720")
        self.root.configure(bg="#121212")

        # UI変数
        self.guide_opacity = tk.DoubleVar(value=50.0)
        self.guide_visible = tk.BooleanVar(value=True)
        self.ink_opacity = tk.DoubleVar(value=90.0)
        self.nijimi_speed = tk.DoubleVar(value=1.0)
        self.model_text = tk.StringVar(value="菜々芭那々（Nanobanana）")
        self.models = ["菜々芭那々（Nanobanana）", "永", "道", "未来", "心", "和", "龍"]

        self.draw_mode = tk.StringVar(value="Hand Mode")
        self.modes = ["Hand Mode", "Body Mode", "Head Mode"]

        # ONNXエンジン初期化
        self.estimator = WholeBodyEstimator(
            model_path="yolox_s_wholebody12_post_0190_1x3x480x960.onnx"
        )

        self.current_brush_size = 30
        self.target_detected = False

        # カメラ設定
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ret, frame = self.cap.read()
        if ret:
            self.H, self.W = frame.shape[:2]
        else:
            self.H, self.W = 720, 1280
            print("Warning: webcam not accessible.")

        # 墨キャンバス
        self.ink_mask = np.zeros((self.H, self.W), dtype=np.uint8)

        # 描画ステータス
        self.is_drawing = False
        self.last_x, self.last_y = 0, 0
        self.stationary_x, self.stationary_y = 0, 0
        self.stationary_start_time = 0
        self.BLEED_DELAY = 0.5
        self.BLEED_TOLERANCE = 50

        self.setup_ui()
        self.update_guide_text()
        self.root.after(10, self.update_frame)

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='#e0e0e0')
        style.configure('TButton', background='#bb86fc', foreground='#000000')
        style.configure('TCheckbutton', background='#1e1e1e', foreground='#e0e0e0')

        # 左サイドバー
        left_frame = tk.Frame(self.root, bg="#1e1e1e", width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="System Settings", bg="#1e1e1e", fg="#fff",
                 font=("Arial", 14, "bold")).pack(pady=20, padx=10, anchor="w")

        tk.Label(left_frame, text="Draw Mode (描画モード)",
                 bg="#1e1e1e", fg="#e0e0e0").pack(anchor="w", padx=20, pady=(10, 0))
        ttk.Combobox(left_frame, textvariable=self.draw_mode,
                     values=self.modes, state="readonly").pack(fill=tk.X, padx=20, pady=5)

        tk.Button(left_frame, text="Reset Canvas", bg="#cf6679", fg="#000",
                  font=("Arial", 10, "bold"), command=self.reset_canvas,
                  relief="flat", padx=10, pady=5).pack(fill=tk.X, padx=20, pady=10)

        tk.Button(left_frame, text="Save Image", bg="#03dac6", fg="#000",
                  font=("Arial", 10, "bold"), command=self.save_image,
                  relief="flat", padx=10, pady=5).pack(fill=tk.X, padx=20, pady=10)

        tk.Checkbutton(left_frame, text="Guide Toggle (表示/非表示)",
                       variable=self.guide_visible, bg="#1e1e1e", fg="#e0e0e0",
                       selectcolor="#333").pack(anchor="w", padx=20, pady=10)

        tk.Label(left_frame, text="Guide Opacity (お手本)",
                 bg="#1e1e1e", fg="#e0e0e0").pack(anchor="w", padx=20, pady=(10, 0))
        tk.Scale(left_frame, variable=self.guide_opacity, from_=0, to=100,
                 orient=tk.HORIZONTAL, bg="#1e1e1e", fg="#e0e0e0",
                 highlightthickness=0, bd=0).pack(fill=tk.X, padx=20, pady=5)

        tk.Label(left_frame, text="Ink Opacity (墨の濃さ)",
                 bg="#1e1e1e", fg="#e0e0e0").pack(anchor="w", padx=20, pady=(10, 0))
        tk.Scale(left_frame, variable=self.ink_opacity, from_=10, to=100,
                 orient=tk.HORIZONTAL, bg="#1e1e1e", fg="#e0e0e0",
                 highlightthickness=0, bd=0).pack(fill=tk.X, padx=20, pady=5)

        # 右サイドバー
        right_frame = tk.Frame(self.root, bg="#1e1e1e", width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="Model & Effects", bg="#1e1e1e", fg="#fff",
                 font=("Arial", 14, "bold")).pack(pady=20, padx=10, anchor="w")

        tk.Label(right_frame, text="Model Text (文字)",
                 bg="#1e1e1e", fg="#e0e0e0").pack(anchor="w", padx=20, pady=(10, 0))
        combo = ttk.Combobox(right_frame, textvariable=self.model_text,
                             values=self.models, state="readonly")
        combo.pack(fill=tk.X, padx=20, pady=5)
        combo.bind("<<ComboboxSelected>>", self.update_guide_text)

        tk.Label(right_frame, text="Nijimi Speed (にじみ)",
                 bg="#1e1e1e", fg="#e0e0e0").pack(anchor="w", padx=20, pady=(10, 0))
        tk.Scale(right_frame, variable=self.nijimi_speed, from_=0.1, to=5.0,
                 resolution=0.1, orient=tk.HORIZONTAL, bg="#1e1e1e", fg="#e0e0e0",
                 highlightthickness=0, bd=0).pack(fill=tk.X, padx=20, pady=5)

        # 映像エリア
        center_frame = tk.Frame(self.root, bg="#000000")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(center_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def get_font(self, size):
        system = platform.system()
        fonts = (["msgothic.ttc", "msmincho.ttc", "meiryo.ttc", "yumin.ttf", "arial.ttf"]
                 if system == "Windows"
                 else ["NotoSerifCJK-Regular.ttc", "fonts-japanese-mincho.ttf", "Arial.ttf"])
        for f in fonts:
            try:
                return ImageFont.truetype(f, size)
            except IOError:
                continue
        return ImageFont.load_default()

    def update_guide_text(self, *args):
        text = self.model_text.get()
        if "Nanobanana" in text:
            text = "菜々芭那々"

        img = Image.new("L", (self.W, self.H), 0)
        draw = ImageDraw.Draw(img)
        font_size = min(int(self.H * 0.8), int(self.W * 0.8 / max(1, len(text))))
        font = self.get_font(font_size)

        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y = (self.W - tw) // 2 - bbox[0], (self.H - th) // 2 - bbox[1]
        else:
            tw, th = draw.textsize(text, font=font)
            x, y = (self.W - tw) // 2, (self.H - th) // 2

        draw.text((x, y), text, font=font, fill=255)
        self.guide_mask = np.array(img, dtype=np.uint8)

    def reset_canvas(self):
        self.ink_mask.fill(0)

    def save_image(self):
        if hasattr(self, 'current_composed_frame'):
            path = filedialog.asksaveasfilename(
                defaultextension=".png", filetypes=[("PNG Files", "*.png")]
            )
            if path:
                cv2.imwrite(path, self.current_composed_frame)
                messagebox.showinfo("Saved", f"Image saved to {path}")

    def draw_line(self, x1, y1, x2, y2, thickness):
        color = int(self.ink_opacity.get() * 2.55)
        cv2.line(self.ink_mask, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    def apply_bleed(self, cx, cy, thickness):
        now = time.time()
        if not hasattr(self, 'last_bleed_time'):
            self.last_bleed_time = now
        interval = 0.05 / max(0.1, self.nijimi_speed.get())
        if now - self.last_bleed_time < interval:
            return
        self.last_bleed_time = now

        r = int(thickness) + 40
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(self.W, cx + r), min(self.H, cy + r)
        if x2 <= x1 or y2 <= y1:
            return

        roi = self.ink_mask[y1:y2, x1:x2]
        if np.max(roi) == 0:
            return

        k_size = max(3, int(3 * self.nijimi_speed.get()))
        k_size += 1 if k_size % 2 == 0 else 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(roi, kernel, iterations=1)
        blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
        self.ink_mask[y1:y2, x1:x2] = np.maximum(roi, blurred)

    def process_tracking(self, frame_bgr):
        """
        バウンディングボックスの中心を描画点、サイズを筆の太さに使う。
        """
        mode = self.draw_mode.get()
        x, y, thickness = 0, 0, 30
        detected = False

        detections = self.estimator.infer(frame_bgr)
        if detections is None:
            return detected, x, y, thickness

        if mode == "Hand Mode":
            box = detections["hand"]
            if box is None:
                return detected, x, y, thickness
            x, y = int(box["cx"]), int(box["cy"])
            # ボックスの短辺を筆の太さに使う（手が大きい=近い=太い筆）
            thickness = max(10, min(150, int(min(box["w"], box["h"]) * 0.3)))

        elif mode == "Body Mode":
            box = detections["body"]
            if box is None:
                return detected, x, y, thickness
            x, y = int(box["cx"]), int(box["cy"])
            # 体の幅を筆の太さに使う
            thickness = max(20, min(300, int(box["w"] * 0.15)))

        elif mode == "Head Mode":
            # head → face の順でフォールバック
            box = detections["head"] or detections["face"]
            if box is None:
                return detected, x, y, thickness
            x, y = int(box["cx"]), int(box["cy"])
            # 顔の幅（カメラに近いほど太くなる）
            thickness = max(10, min(250, int(box["w"] * 0.4)))

        detected = True
        return detected, x, y, thickness

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.W, self.H))
        else:
            frame = np.ones((self.H, self.W, 3), dtype=np.uint8) * 40

        # BGRのままinferに渡す（RGB変換不要）
        self.target_detected, hx, hy, self.current_brush_size = \
            self.process_tracking(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.target_detected:
            if 0 <= hx < self.W and 0 <= hy < self.H:
                if not self.is_drawing:
                    self.is_drawing = True
                    self.last_x, self.last_y = hx, hy
                    self.stationary_x, self.stationary_y = hx, hy
                    self.stationary_start_time = time.time()

                dist = math.hypot(hx - self.stationary_x, hy - self.stationary_y)
                if dist > self.BLEED_TOLERANCE:
                    self.stationary_x, self.stationary_y = hx, hy
                    self.stationary_start_time = time.time()

                self.draw_line(self.last_x, self.last_y, hx, hy, self.current_brush_size)
                self.last_x, self.last_y = hx, hy

                if time.time() - self.stationary_start_time >= self.BLEED_DELAY:
                    self.apply_bleed(self.stationary_x, self.stationary_y, self.current_brush_size)
            else:
                self.is_drawing = False
        else:
            self.is_drawing = False

        # 十字線
        cx_l, cy_l = self.W // 2, self.H // 2
        cv2.line(frame_rgb, (cx_l, 0), (cx_l, self.H), (255, 255, 255), 1)
        cv2.line(frame_rgb, (0, cy_l), (self.W, cy_l), (255, 255, 255), 1)

        # お手本レイヤー合成
        if self.guide_visible.get() and self.guide_mask is not None:
            opacity = self.guide_opacity.get() / 100.0
            if opacity > 0:
                alpha_3d = (self.guide_mask / 255.0)[:, :, np.newaxis] * opacity
                frame_rgb = frame_rgb * (1 - alpha_3d) + 255 * alpha_3d

        # 墨レイヤー合成
        ink_alpha = (self.ink_mask / 255.0)[:, :, np.newaxis]
        frame_rgb = frame_rgb * (1 - ink_alpha)

        # 筆カーソル
        if self.target_detected:
            cv2.circle(frame_rgb, (hx, hy), self.current_brush_size // 2, (255, 0, 0), 2)
            cv2.circle(frame_rgb, (hx, hy), 3, (255, 0, 0), -1)

        frame_rgb = frame_rgb.astype(np.uint8)
        self.current_composed_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # ステータス表示
        status_color = (0, 255, 0) if self.target_detected else (255, 0, 0)
        status_text = (f"ONNX Active ({self.draw_mode.get()})"
                       if self.target_detected else "No Person Detected")
        cv2.putText(frame_rgb, f"Status: {status_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame_rgb, f"Brush Size: {self.current_brush_size}px", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        lw = self.video_label.winfo_width()
        lh = self.video_label.winfo_height()
        display = cv2.resize(frame_rgb, (lw, lh)) if lw > 10 else frame_rgb
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(display))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_frame)

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.estimator.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitalCalligraphyApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()