with open("sssss.py", "r") as f:
    content = f.read()

content = content.replace(
    "from tkinter import ttk, filedialog, messagebox",
    "from tkinter import ttk, filedialog, messagebox, simpledialog"
)

content = content.replace(
    "import math\n\n# wholebody12 クラスID定義",
    "import math\nimport requests\nimport base64\nimport threading\nimport json\n\n# wholebody12 クラスID定義"
)

content = content.replace(
    "# 墨キャンバス\n        self.ink_mask = np.zeros((self.H, self.W), dtype=np.uint8)\n\n        # 描画ステータス",
    "# 墨キャンバス\n        self.ink_mask = np.zeros((self.H, self.W), dtype=np.uint8)\n\n        # カメラ設定用変数\n        self.camera_id_var = tk.StringVar(value=\"0\")\n        self.camera_res_var = tk.StringVar(value=\"1280x720\")\n\n        # 手の開き・グー判定ステータス\n        self.max_hand_ratio = 0.0\n        self.is_fist = False\n        self.fist_threshold = 0.65\n\n        # デバッグ画面 (裏画面) 用\n        self.debug_window = None\n        self.debug_canvas = None\n        self.debug_text = None\n\n        # 描画ステータス"
)

with open("sssss.py", "w") as f:
    f.write(content)
