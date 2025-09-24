import os, sys, time, json, hashlib, socket, uuid, platform
from pathlib import Path
from datetime import datetime
from urllib import request as urlreq
from urllib.error import URLError, HTTPError

import cv2

# Try to import Tk only if needed (first run)
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# =================== SETTINGS ===================
APP_VERSION = "1.1.0"
REQUIRED_CONSEC = 3        # frames with face before we accept "present"
TIMEOUT_SECS = 25          # max wait for a face before giving up
OUTPUT_DIR = "captures"    # where to save images
CAM_INDEX = 0              # change if multiple cameras
CONF_THR = 0.5             # DNN confidence threshold (0.5 ~ 0.7)
# DNN model files (if absent, auto-fallback to Haar)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PROTO = BASE_DIR / "deploy.prototxt"
MODEL_WEIGHTS = BASE_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
# Permanent profile path
PROFILE_PATH = Path.home() / ".face_attendance" / "profile.json"
# =================================================

# ----------- EXPORTS YOU CAN SEND ---------------
SENDER_READY_PAYLOAD = None   # dict
PAYLOAD_JSON = None           # JSON string of dict above
LAST_SAVED_IMAGE_PATH = None  # Path or None
# ------------------------------------------------

MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]

def format_pretty_now(dt: datetime) -> str:
    # "7 Sept, 7:00 am"
    day = str(dt.day)
    month = MONTH_ABBR[dt.month - 1]
    hour_12 = dt.strftime("%I").lstrip("0") or "0"
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p").lower()
    return f"{day} {month}, {hour_12}:{minute} {ampm}"

def safe_filename(s: str) -> str:
    keep = "-_.() "
    return "".join(c if c.isalnum() or c in keep else "_" for c in s)

def load_profile():
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_profile(name: str, seat: str):
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump({"name": name, "seat": seat}, f, ensure_ascii=False, indent=2)
    print(f'[profile] Saved permanent profile at "{PROFILE_PATH}"', flush=True)

def _public_ip(timeout=2.5):
    try:
        with urlreq.urlopen("https://api.ipify.org?format=json", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("ip")
    except Exception:
        return None

def _geo_from_ip(timeout=2.5):
    try:
        with urlreq.urlopen("https://ipapi.co/json/", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
            return {
                "country": data.get("country_name"),
                "region":  data.get("region"),
                "city":    data.get("city"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "org":     data.get("org"),
                "timezone": data.get("timezone"),
            }
    except Exception:
        return None

def _local_ip_guess():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None

def collect_device_info():
    tz = str(datetime.now().astimezone().tzinfo)
    try:
        uname = platform.uname()
    except Exception:
        uname = None
    info = {
        "device_id": f"{uuid.getnode():012x}",
        "hostname": socket.gethostname(),
        "local_ip": _local_ip_guess(),
        "public_ip": _public_ip(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "timezone": tz,
        "app_version": APP_VERSION,
    }
    if uname:
        info.update({
            "platform_node": uname.node,
            "platform_system": uname.system,
            "platform_release": uname.release,
            "platform_version": uname.version,
        })
    geo = _geo_from_ip()
    if geo:
        info["geo"] = geo
    return info

class OneShotDetector:
    """Open camera, wait for stable face, save once, then exit."""
    def __init__(self):
        self.use_dnn = MODEL_PROTO.exists() and MODEL_WEIGHTS.exists()
        self.net = None
        self.haar = None
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def _load(self):
        if self.use_dnn:
            try:
                self.net = cv2.dnn.readNetFromCaffe(str(MODEL_PROTO), str(MODEL_WEIGHTS))
                print("[detector] Using DNN face detector", flush=True)
                return
            except Exception as e:
                print(f"[warn] DNN load failed: {e} → fallback to Haar", file=sys.stderr)
                self.use_dnn = False
        # Haar fallback
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)
        if self.haar.empty():
            raise RuntimeError("Failed to load any face detector.")
        print("[detector] Using Haar face detector", flush=True)

    def _detect(self, frame_bgr):
        if self.use_dnn and self.net is not None:
            (h, w) = frame_bgr.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0)
            )
            self.net.setInput(blob)
            dets = self.net.forward()
            boxes = []
            for i in range(dets.shape[2]):
                conf = float(dets[0, 0, i, 2])
                if conf >= CONF_THR:
                    (x1, y1, x2, y2) = (dets[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
            return boxes
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    def capture_once(self, name: str, seat: str):
        """
        Returns: (code, out_path, saved_at_iso, detector_used)
          code: 0 saved, 1 timed out (no face), 2 camera/error
        """
        name_sf = safe_filename(name.strip())
        seat_sf = safe_filename(seat.strip())
        if not name_sf or not seat_sf:
            print("[error] Empty name/seat.", file=sys.stderr)
            return 2, None, None, None

        # Load detector
        self._load()

        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            print("[fatal] Could not open webcam.", file=sys.stderr)
            return 2, None, None, None

        start = time.time()
        streak = 0
        detector_used = "dnn" if self.use_dnn else "haar"
        out_path, saved_at_iso = None, None

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.03)
                    if (time.time() - start) > TIMEOUT_SECS:
                        print("Device opened without face", flush=True)
                        return 1, None, None, detector_used
                    continue

                boxes = self._detect(frame)
                if len(boxes) > 0:
                    streak += 1
                else:
                    streak = 0

                if streak >= REQUIRED_CONSEC:
                    print("present", flush=True)
                    pretty = format_pretty_now(datetime.now())
                    fname = f"{name_sf}_{seat_sf}_{safe_filename(pretty)}.jpg"
                    out_path = Path(OUTPUT_DIR) / fname
                    ok = cv2.imwrite(str(out_path), frame)
                    if not ok:
                        print("[save_error] cv2.imwrite failed.", file=sys.stderr)
                        return 2, None, None, detector_used
                    saved_at_iso = datetime.now().astimezone().isoformat()
                    print(f"[saved] {out_path}", flush=True)
                    return 0, str(out_path), saved_at_iso, detector_used

                if (time.time() - start) > TIMEOUT_SECS:
                    print("Device opened without face", flush=True)
                    return 1, None, None, detector_used

                time.sleep(0.03)
        finally:
            if cap and cap.isOpened():
                cap.release()

# ----------------- First-run Tiny Form -----------------
class FirstRunForm:
    """Ask Name + Seat; buttons: Start (Temporary) and Permanent. Captures once then exits."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("User Details")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        frm = ttk.Frame(self.root, padding=16)
        frm.grid(row=0, column=0)

        ttk.Label(frm, text="Name").grid(row=0, column=0, sticky="w", pady=(0,8))
        self.name_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.name_var, width=30).grid(row=0, column=1, pady=(0,8))

        ttk.Label(frm, text="Seat No.").grid(row=1, column=0, sticky="w", pady=(0,8))
        self.seat_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.seat_var, width=30).grid(row=1, column=1, pady=(0,8))

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=2, pady=(12,0), sticky="ew")

        self.btn_temp = ttk.Button(btns, text="Start (Temporary)", command=self.start_temp)
        self.btn_temp.grid(row=0, column=0, padx=(0,6), sticky="ew")
        self.btn_perm = ttk.Button(btns, text="Permanent", command=self.start_perm)
        self.btn_perm.grid(row=0, column=1, sticky="ew")

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

    def _finalize_payload_and_exit(self, name, seat, code, out_path, saved_at_iso, detector_used):
        global SENDER_READY_PAYLOAD, PAYLOAD_JSON, LAST_SAVED_IMAGE_PATH
        device = collect_device_info()

        img_sha256 = None
        img_size = None
        if out_path and os.path.exists(out_path):
            LAST_SAVED_IMAGE_PATH = out_path
            try:
                with open(out_path, "rb") as f:
                    img_bytes = f.read()
                    img_sha256 = hashlib.sha256(img_bytes).hexdigest()
                    img_size = len(img_bytes)
            except Exception:
                pass

        status = {0: "saved", 1: "no_face", 2: "error"}.get(code, "unknown")

        SENDER_READY_PAYLOAD = {
            "status": status,
            "user": {
                "name": name,
                "seat": seat,
            },
            "capture": {
                "image_path": out_path,
                "saved_at_iso": saved_at_iso,
                "saved_at_human": format_pretty_now(datetime.now()),
                "image_sha256": img_sha256,
                "image_size_bytes": img_size,
            },
            "runtime": {
                "detector": detector_used,
                "cam_index": CAM_INDEX,
                "timeout_secs": TIMEOUT_SECS,
                "required_consecutive_frames": REQUIRED_CONSEC,
                "app_version": APP_VERSION,
            },
            "device": device,
        }
        PAYLOAD_JSON = json.dumps(SENDER_READY_PAYLOAD, ensure_ascii=False)

        # Optional: print payload JSON so you can capture logs or pipe it
        print("PAYLOAD_JSON:", PAYLOAD_JSON, flush=True)

        # Example (commented): send to webhook/API
        # import requests
        # requests.post("https://your.webhook/endpoint", json=SENDER_READY_PAYLOAD, timeout=5)

        self.root.destroy()
        sys.exit(code)

    def _run_capture(self, name: str, seat: str):
        now = format_pretty_now(datetime.now())
        print(f'USER_DETAILS: name="{name}", seat="{seat}", submitted_at="{now}"', flush=True)
        self.root.withdraw()  # hide UI while capturing
        code, out_path, saved_at_iso, detector_used = OneShotDetector().capture_once(name, seat)
        # Build payload and exit with the exact result code
        self._finalize_payload_and_exit(name, seat, code, out_path, saved_at_iso, detector_used)

    def start_temp(self):
        name = self.name_var.get().strip()
        seat = self.seat_var.get().strip()
        if not name or not seat:
            messagebox.showwarning("Missing info", "Please enter both Name and Seat No.")
            return
        self._run_capture(name, seat)

    def start_perm(self):
        name = self.name_var.get().strip()
        seat = self.seat_var.get().strip()
        if not name or not seat:
            messagebox.showwarning("Missing info", "Please enter both Name and Seat No.")
            return
        save_profile(name, seat)
        self._run_capture(name, seat)

    def on_close(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# ----------------------- Main -----------------------
def main():
    global SENDER_READY_PAYLOAD, PAYLOAD_JSON, LAST_SAVED_IMAGE_PATH

    # Allow clearing saved profile
    if "--reset-profile" in sys.argv:
        try:
            if PROFILE_PATH.exists():
                PROFILE_PATH.unlink()
                print("[profile] Removed permanent profile.")
        except Exception as e:
            print(f"[profile] Remove failed: {e}", file=sys.stderr)

    prof = load_profile()
    if prof and prof.get("name") and prof.get("seat"):
        # Headless mode: capture once and exit
        name = prof["name"].strip()
        seat = prof["seat"].strip()
        print(f'[profile] Loaded permanent profile for {name}/{seat}. Running headless...', flush=True)
        code, out_path, saved_at_iso, detector_used = OneShotDetector().capture_once(name, seat)

        # Build payload before exiting
        device = collect_device_info()
        img_sha256 = None
        img_size = None
        if out_path and os.path.exists(out_path):
            LAST_SAVED_IMAGE_PATH = out_path
            try:
                with open(out_path, "rb") as f:
                    img_bytes = f.read()
                    img_sha256 = hashlib.sha256(img_bytes).hexdigest()
                    img_size = len(img_bytes)
            except Exception:
                pass

        status = {0: "saved", 1: "no_face", 2: "error"}.get(code, "unknown")
        SENDER_READY_PAYLOAD = {
            "status": status,
            "user": {"name": name, "seat": seat},
            "capture": {
                "image_path": out_path,
                "saved_at_iso": saved_at_iso,
                "saved_at_human": format_pretty_now(datetime.now()),
                "image_sha256": img_sha256,
                "image_size_bytes": img_size,
            },
            "runtime": {
                "detector": detector_used,
                "cam_index": CAM_INDEX,
                "timeout_secs": TIMEOUT_SECS,
                "required_consecutive_frames": REQUIRED_CONSEC,
                "app_version": APP_VERSION,
            },
            "device": device,
        }
        PAYLOAD_JSON = json.dumps(SENDER_READY_PAYLOAD, ensure_ascii=False)
        print("PAYLOAD_JSON:", PAYLOAD_JSON, flush=True)

        # Example (commented): POST it
        # import requests
        # requests.post("https://your.webhook/endpoint", json=SENDER_READY_PAYLOAD, timeout=5)

        sys.exit(code)

    # No profile: need GUI once
    if not TK_AVAILABLE:
        print("[fatal] No permanent profile and Tkinter not available.", file=sys.stderr)
        print("Install Tkinter or run once on a machine with GUI to save profile:", file=sys.stderr)
        print("  sudo apt-get install python3-tk", file=sys.stderr)
        sys.exit(3)

    FirstRunForm().run()

if __name__ == "__main__":
    main()
