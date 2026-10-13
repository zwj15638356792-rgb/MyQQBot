"""双击入口的扫码窗口。仅启动本机 NapCat 和本项目，不发送测试消息。"""

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from dotenv import dotenv_values


PROJECT = Path(__file__).resolve().parent


class StartupError(Exception):
    pass


class ServiceNotReady(StartupError):
    pass


def is_admin():
    import ctypes
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def request_admin():
    """由 Windows 弹出 UAC，路径通过独立参数传递，不拼接 shell 命令。"""
    import ctypes
    execute = ctypes.windll.shell32.ShellExecuteW
    execute.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                        ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int]
    execute.restype = ctypes.c_void_p
    arguments = subprocess.list2cmdline([str(PROJECT / "start_bot.py"), "--elevated"])
    return execute(None, "runas", sys.executable, arguments, str(PROJECT), 1) or 0


def ensure_admin():
    """提升权限在获取单实例锁之前完成，取消 UAC 时不启动任何服务。"""
    if os.name != "nt" or is_admin():
        return True
    if "--elevated" in sys.argv:
        raise StartupError("没有获得管理员权限，请右键“一键启动.cmd”，选择“以管理员身份运行”。")
    if request_admin() <= 32:
        raise StartupError("未获得管理员权限，启动已取消。请重新双击启动，并在 Windows 提示中点“是”。")
    return False  # 新的管理员进程继续启动，当前进程退出。


class LocalAPI:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        # 本机服务不经过系统代理，避免代理软件影响扫码与连接。
        self.opener = build_opener(ProxyHandler({}))

    def post(self, path, body=None, credential=""):
        headers = {"Content-Type": "application/json"}
        if credential:
            headers["Authorization"] = "Bearer " + credential
        request = Request(self.base_url + path, data=json.dumps(body or {}).encode(),
                          headers=headers, method="POST")
        with self.opener.open(request, timeout=3) as response:
            return json.loads(response.read())


class NapCatAPI(LocalAPI):
    def __init__(self, root):
        self.config_path = root / "napcat/config/webui.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        if self.config.get("disableWebUI"):
            raise StartupError("NapCat 的扫码服务已关闭，请开启 WebUI 后再试。")
        super().__init__(f"http://127.0.0.1:{int(self.config.get('port', 6099))}/api")
        self.credential = ""

    def authenticate(self):
        # Token 只从本机配置读取，不放在浏览器地址或启动参数中。
        self.config = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        password_hash = hashlib.sha256((self.config["token"] + ".napcat").encode()).hexdigest()
        packet = self.post("/auth/login", {"hash": password_hash})
        data = packet.get("data") or {}
        if data.get("require2FA"):
            raise StartupError("NapCat 开启了二次验证，需要先在它的管理页面完成登录。")
        if not data.get("Credential"):
            raise StartupError("无法连接 NapCat 扫码服务，请检查它的本机登录配置。")
        self.credential = data["Credential"]

    def call(self, path):
        if not self.credential:
            self.authenticate()
        packet = self.post(path, credential=self.credential)
        if packet.get("code") != 0:
            self.credential = ""
            raise ServiceNotReady("NapCat 暂时没有准备好，请稍候。")
        return packet.get("data") or {}


def existing_bot(project):
    """兼容手工启动的 main.py，避免同时跑两份机器人。"""
    import psutil

    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if not (process.info["name"] or "").lower().startswith("python"):
                continue
            for argument in (process.info["cmdline"] or [])[1:]:
                if Path(argument).name.lower() != "main.py":
                    continue
                target = Path(argument)
                if not target.is_absolute():
                    target = Path(process.cwd()) / target
                if target.resolve() == (project / "main.py").resolve():
                    return process.pid
        except (psutil.Error, OSError, ValueError):
            continue
    return None


class Launcher:
    def __init__(self, project=PROJECT):
        self.project = project
        self.settings = json.loads((project / "launcher_settings.json").read_text(encoding="utf-8-sig"))
        self.napcat_dir = Path(self.settings["napcat_dir"])
        self.python_exe = Path(self.settings["python_exe"])
        self.account = str(self.settings["account_id"])
        self.runtime = project / ".tmp/launcher"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.refresh = threading.Event()
        self.napcat = None
        self.bot = None
        self.reused_bot_pid = None
        self.log_handles = []
        self.qr_service_started = False
        self.qr_before_start = None
        self.api = None
        self.env = {**dotenv_values(project / ".env"), **os.environ}
        self.last_status = None

    def emit(self, kind, value):
        if kind == "status" and value == self.last_status:
            return
        if kind == "status":
            self.last_status = value
        self.events.put((kind, value))

    def validate(self):
        if not all((self.napcat_dir / file).is_file() for file in ("napcat.bat", "node.exe", "index.js")):
            raise StartupError("没有找到 NapCat。请检查 launcher_settings.json 中的安装路径。")
        if not self.python_exe.is_file():
            raise StartupError("没有找到 Python。请检查 launcher_settings.json 中的安装路径。")
        config_path = self.napcat_dir / f"napcat/config/onebot11_{self.account}.json"
        if not config_path.is_file():
            raise StartupError("这个 QQ 还没有 NapCat 连接配置，请先完成首次配置。")
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        for kind, url_key, token_key in (
                ("httpServers", "HTTP_URL", "NAPCAT_TOKEN"),
                ("websocketServers", "WS_URL", "NAPCAT_WS_TOKEN")):
            url = urlsplit(self.env.get(url_key, ""))
            if url.hostname not in ("127.0.0.1", "localhost", "::1"):
                raise StartupError("一键扫码启动适用于本机 NapCat，请把连接地址配置为本机地址。")
            token = self.env.get(token_key) or self.env.get("NAPCAT_TOKEN")
            matched = any(item.get("enable") and int(item.get("port", 0)) == url.port
                          and item.get("token") == token
                          for item in config.get("network", {}).get(kind, []))
            if not matched:
                raise StartupError("NapCat 与机器人连接配置不一致，请检查端口和 Token。")
        self.api = NapCatAPI(self.napcat_dir)

    def qr_signature(self):
        path = self.napcat_dir / "napcat/cache/qrcode.png"
        try:
            return (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            return None

    def ready_qr_path(self):
        """只有登录服务已运行且冷启动后图片已重写，才显示缓存二维码。"""
        if not self.qr_service_started:
            return None
        signature = self.qr_signature()
        if signature is None or signature == self.qr_before_start:
            return None
        return self.napcat_dir / "napcat/cache/qrcode.png"

    def start_napcat(self):
        self.qr_before_start = self.qr_signature()
        # Windows 的 bat 必须交给 cmd 执行；使用完整引号支持安装目录中的空格和括号。
        command = f'"{os.environ.get("COMSPEC", "C:/Windows/System32/cmd.exe")}" /d /s /c ""{self.napcat_dir / "napcat.bat"}""'
        self.napcat = self.spawn(command, self.napcat_dir, "napcat.log")
        self.qr_service_started = True

    def spawn(self, arguments, cwd, log_name):
        if self.stop.is_set():
            raise StartupError("启动已取消。")
        output = open(self.runtime / log_name, "ab", buffering=0)
        self.log_handles.append(output)
        return subprocess.Popen(arguments, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})

    def onebot_ready(self):
        packet = LocalAPI(self.env["HTTP_URL"]).post("/get_login_info", credential=self.env["NAPCAT_TOKEN"])
        if packet.get("status") != "ok" or packet.get("retcode") != 0:
            return False
        return str((packet.get("data") or {}).get("user_id")) == self.account

    def run(self):
        try:
            self.validate()
            self.emit("status", "正在准备扫码登录…")
            try:
                self.api.authenticate()
                self.qr_service_started = True  # 已有登录服务时复用，不重复运行 bat。
            except HTTPError as exc:
                raise StartupError("NapCat 扫码服务拒绝连接，请检查本机 WebUI 配置。") from exc
            except (URLError, TimeoutError, ConnectionError):
                self.emit("status", "正在启动 QQ 登录服务…")
                self.start_napcat()
            failures_since = time.monotonic()
            qr_missing_since = None
            while not self.stop.is_set():
                if self.napcat is not None and self.napcat.poll() is not None:
                    raise StartupError("QQ 登录服务已退出，请点“查看日志”检查原因。")
                if self.bot is not None and self.bot.poll() is not None:
                    raise StartupError("机器人已退出，请点“查看日志”检查原因。")
                try:
                    if self.refresh.is_set():
                        self.refresh.clear()
                        self.api.call("/QQLogin/RefreshQRcode")
                    state = self.api.call("/QQLogin/CheckLoginStatus")
                    failures_since = time.monotonic()
                    if state.get("isLogin"):
                        info = self.api.call("/QQLogin/GetQQLoginInfo")
                        if str(info.get("uin")) != self.account:
                            raise StartupError("扫码登录的 QQ 不是配置的机器人账号，请使用机器人 QQ 扫码。")
                        self.emit("qr", None)
                        if self.bot is None and self.reused_bot_pid is None:
                            self.emit("status", "登录成功，正在连接机器人…")
                            if not self.onebot_ready():
                                self.stop.wait(1.5)
                                continue
                            pid = existing_bot(self.project)
                            if pid is None:
                                self.bot = self.spawn([str(self.python_exe), "-u", "main.py"],
                                                      self.project, "bot.log")
                            else:
                                self.reused_bot_pid = pid
                                self.emit("existing", pid)
                        self.emit("status", "已登录，机器人正在运行。现在可以发消息了。")
                        self.emit("online", True)
                    else:
                        self.emit("online", False)
                        qr_path = self.ready_qr_path()
                        if state.get("qrLoginAccepted"):
                            qr_missing_since = None
                            self.emit("status", "已扫码，请在手机上确认登录。")
                        elif qr_path is not None:
                            qr_missing_since = None
                            self.emit("status", "请用机器人 QQ 扫描下方二维码，并在手机上确认登录。")
                            self.emit("qr", qr_path)
                        else:
                            self.emit("status", "登录服务已启动，正在等待新二维码…")
                            if qr_missing_since is None:
                                qr_missing_since = time.monotonic()
                            elif time.monotonic() - qr_missing_since > 60:
                                raise StartupError("登录服务没有更新二维码。请关闭本窗口，关闭原来手工启动的 NapCat，再双击一键启动，并在管理员提示中点“是”。仍失败可点“查看日志”。")
                except (URLError, TimeoutError, ConnectionError, ServiceNotReady):
                    self.api.credential = ""
                    self.emit("status", "正在等待 QQ 登录服务准备好…")
                    if time.monotonic() - failures_since > 90:
                        raise StartupError("QQ 登录服务长时间没有响应，请点“查看日志”检查原因。")
                self.stop.wait(1.5)
        except Exception as exc:
            # 第三方原始错误可能含登录信息，窗口只显示可控的提示。
            message = str(exc) if isinstance(exc, StartupError) else f"启动失败（{type(exc).__name__}），请查看日志。"
            self.emit("error", message)

    def close(self):
        self.stop.set()
        # 只关闭这个启动器启动的进程，保留用户原来已运行的实例。
        for process in (self.bot, self.napcat):
            if process is not None and process.poll() is None:
                # bat 的 cmd 进程下面还有 node，关闭时一并清理自己启动的子进程。
                descendants = []
                if isinstance(process.pid, int):
                    import psutil
                    try:
                        descendants = psutil.Process(process.pid).children(recursive=True)
                    except psutil.Error:
                        pass
                for child in reversed(descendants):
                    try:
                        child.terminate()
                    except psutil.Error:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
        for handle in self.log_handles:
            handle.close()
        self.log_handles.clear()


def main():
    try:
        if not ensure_admin():
            return
    except StartupError as exc:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, str(exc), "QQ机器人 · 一键启动", 0x10)
        return

    import msvcrt
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk

    os.chdir(PROJECT)
    try:
        with open(PROJECT / "persona.toml", "rb") as source:
            name = tomllib.load(source).get("name", "QQ机器人")
    except (OSError, ValueError):
        name = "QQ机器人"
    title = f"{name} · 一键启动"
    runtime = PROJECT / ".tmp/launcher"
    runtime.mkdir(parents=True, exist_ok=True)
    lock = open(runtime / "launcher.lock", "a+b")
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 9)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        lock.close()
        return

    window = tk.Tk()
    window.title(title)
    window.geometry("520x680")
    window.resizable(False, False)
    window.configure(bg="#f5f7fb")
    tk.Label(window, text=name, bg="#f5f7fb", fg="#1f2937",
             font=("Microsoft YaHei UI", 24, "bold")).pack(pady=(28, 5))
    tk.Label(window, text="扫码登录后，机器人会自动启动", bg="#f5f7fb", fg="#64748b",
             font=("Microsoft YaHei UI", 11)).pack()
    status = tk.StringVar(value="正在准备…")
    qr_frame = tk.Frame(window, bg="white", width=320, height=320)
    qr_frame.pack(pady=18)
    qr_frame.pack_propagate(False)
    image_label = tk.Label(qr_frame, text="正在获取二维码…", bg="white",
                           font=("Microsoft YaHei UI", 11))
    image_label.pack(fill="both", expand=True)
    tk.Label(window, textvariable=status, wraplength=460, justify="center", bg="#f5f7fb",
             fg="#334155", font=("Microsoft YaHei UI", 10)).pack(pady=(0, 16))
    buttons = tk.Frame(window, bg="#f5f7fb")
    buttons.pack()
    try:
        launcher = Launcher()
    except Exception:
        messagebox.showerror(title, "无法读取启动设置，请检查 launcher_settings.json。")
        window.destroy()
        lock.close()
        return
    refresh_button = tk.Button(buttons, text="刷新二维码", command=launcher.refresh.set,
                               font=("Microsoft YaHei UI", 10), padx=16, pady=5)
    refresh_button.pack(side="left", padx=7)
    tk.Button(buttons, text="查看日志", command=lambda: os.startfile(str(runtime)),
              font=("Microsoft YaHei UI", 10), padx=16, pady=5).pack(side="left", padx=7)
    tk.Label(window, text="保持此窗口运行。关闭窗口会停止本次启动的机器人和登录服务。",
             wraplength=470, bg="#f5f7fb", fg="#94a3b8", font=("Microsoft YaHei UI", 9)).pack(pady=18)
    qr_stamp = None
    logged_in = False

    def show_qr(path):
        nonlocal qr_stamp
        try:
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            if stamp != qr_stamp:
                with Image.open(path) as original:
                    picture = original.convert("RGB").resize((300, 300), Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(picture)
                image_label.configure(image=photo, text="", width=300, height=300)
                image_label.image = photo
                qr_stamp = stamp
        except OSError:
            pass  # 图片不存在或 NapCat 正在写新二维码，下一轮再读取。

    def update():
        nonlocal qr_stamp, logged_in
        while not launcher.events.empty():
            kind, value = launcher.events.get_nowait()
            if kind in ("status", "error"):
                status.set(value)
                if kind == "error":
                    refresh_button.configure(state="disabled")
                    image_label.configure(image="", text="启动遇到问题\n请查看下方提示", width=28, height=10)
                    image_label.image = None
            elif kind == "qr":
                if value is None:
                    image_label.configure(image="", text="✓\n登录成功", width=28, height=10,
                                          font=("Microsoft YaHei UI", 20), fg="#16a34a")
                    image_label.image = None
                    qr_stamp = None
                else:
                    show_qr(value)
            elif kind == "online":
                logged_in = value
                refresh_button.configure(state="disabled" if value else "normal")
        # 先运行登录服务，再显示它生成的本次二维码，不展示冷启动前的缓存图片。
        if not logged_in:
            ready_path = launcher.ready_qr_path()
            if ready_path is not None:
                show_qr(ready_path)
        window.after(250, update)

    def close():
        launcher.stop.set()
        worker.join(timeout=4)
        launcher.close()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)
    worker = threading.Thread(target=launcher.run, daemon=True)
    worker.start()
    update()
    try:
        window.mainloop()
    finally:
        launcher.close()
        lock.close()


if __name__ == "__main__":
    main()
