"""生图模块：把「角色参考图 + 场景描述」交给图片生成 API，拿回一张同款人物照片。

配置全部读 knowledge.toml（[character] 人物设定 + [image] 参数），
Key 走环境变量 IMAGE_API_KEY。默认对接火山方舟（豆包 Seedream），因为它的
image 字段支持 Base64 参考图、还支持传数组，正好用来做人物一致性。

安全边界：本模块只发「代码拼好的提示词 + 本地参考图」，图片路径由本模块生成，
模型永远碰不到。所以 ai_service.py 里「剥掉模型生成的 CQ 码」那条底线不用为发图让路。

默认关闭。没配 Key 时整个模块静默不工作，不影响现有聊天。
"""

import asyncio
import base64
import binascii
import json
import os
import re
import time
import tomllib
from dataclasses import dataclass, field, fields
from datetime import date, datetime
from pathlib import Path

import aiohttp

try:  # 复用项目统一的日志器，保持 [生图] 标签风格
    from asa.core.logging import get_logger

    logger = get_logger("image")
except Exception:  # 脱离 asabot 单独跑自检时兜底
    import logging

    logger = logging.getLogger("image")

CONFIG_PATH = "knowledge.toml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 场景描述来自群友输入，会原样拼进提示词，所以先做一轮清洗。
# 这只是一道减速带，不是安全保证——真正的底线是"人物描述永远排在前面 + 长度受限"。
MAX_SCENE_CHARS = 120
CQ_CODE = re.compile(r"\[CQ:[^\]]*\]")
SCENE_BLOCKLIST = re.compile(
    r"裸|脱衣|色情|做爱|性行为|血腥|尸体|虐待|自杀|毒品|枪支|爆炸|恐怖分子|"
    r"nude|naked|nsfw|porn|gore|blood",
    re.I,
)

# 配置读不到时的兜底，保证模块不会因为少一行配置就崩
FALLBACK_CHARACTER = {
    "appearance": "年轻东亚女生，20 岁出头，长深棕色微卷发，中分，皮肤白皙，淡妆，五官柔和",
    "outfit": "米白色宽松粗针织毛衣，内搭浅粉色衬衫，米色灯芯绒直筒裤，帆布托特包，米白球鞋",
    "style": "写实照片风格，手机随手拍，自然光，真实皮肤质感，面部和衣服清晰",
    "scene": "室内，白墙，窗边自然光，背景简洁，竖版构图，半身到七分身",
    "ref_dir": "character_refs",
}


class ImageError(Exception):
    """生图链路里任何一种失败。调用方只记日志，不往群里发报错。

    user_safe=True 表示这句话可以直接回给群里（例如"今天额度用完了"）。
    接口报错、网络异常这类属于内部细节，只写日志，群里只回一句笼统的。
    """

    def __init__(self, message, *, user_safe=False):
        super().__init__(message)
        self.user_safe = user_safe


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, int):
        return bool(value)
    return default


def _as_int(value, default, *, minimum=0):
    """把 TOML 里写错类型的值掰回来。写错配置不该让机器人起不来。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        logger.warning("[生图] 配置值 {!r} 不是整数，改用默认 {}", value, default)
        return default
    return number if number >= minimum else default


def _as_float(value, default, *, minimum=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.warning("[生图] 配置值 {!r} 不是数字，改用默认 {}", value, default)
        return default
    return number if number >= minimum else default


def _as_str(value, default=""):
    return value.strip() if isinstance(value, str) and value.strip() else default


def _as_list(value, default, *, keep_empty=False):
    """把配置值掰成字符串数组。

    keep_empty=True 时，显式的空数组是"有意义的空"（例如 allowed_users = []
    表示所有人可用），不能当成"没配"而退回默认值。

    QQ 号在 TOML 里通常写成整数（allowed_users = [10001]），
    所以数字也要收——只认字符串的话这个白名单会被静默清空，等于对所有人开放。
    """
    if isinstance(value, list):
        kept = [str(v).strip() for v in value
                if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip()]
        if kept or keep_empty:
            return kept
        return list(default)
    if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
        return [str(value).strip()]
    return list(default)


@dataclass
class ImageSettings:
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    model: str = "doubao-seedream-4-0-250828"
    size: str = "1728x2304"           # 3:4 竖版，和参考图同比例
    response_format: str = "url"      # url / b64_json，个别服务商只支持其中一种
    max_refs: int = 3
    daily_limit: int = 30             # 每天最多出几张，防刷
    cooldown_seconds: float = 60
    timeout_seconds: float = 120      # 生图比聊天慢得多，超时单独放宽
    output_dir: str = "generated"
    # 参考图上传前缩到长边这么多像素。只用来"认脸"，原始 4K 大图纯属浪费：
    # 3 张 4MB PNG 直接发是 16.6MB 请求体，缩完约 0.8MB。
    # 0 = 不压缩（不建议，容易撞服务端上限）。
    ref_max_edge: int = 1536
    ref_jpeg_quality: int = 88
    # 留空 = 群里任何人都能触发（仍受冷却与每日上限约束）。
    # 填了 QQ 号就只认这些人，防止有人拿它刷额度。
    allowed_users: list = field(default_factory=list)
    commands: list = field(default_factory=lambda: ["画一张", "来个自拍", "生成一张"])
    natural_requests: bool = True    # 识别“帮我画一张”“能发张自拍吗”等明确请求
    context_photos: bool = True      # 无明确场景时，从本次会话提取角色当下的场景

    def __post_init__(self):
        # 配置文件是手写的，类型写错很常见；这里统一掰正，而不是让它在运行中途炸。
        self.enabled = _as_bool(self.enabled)
        self.api_key = _as_str(self.api_key)
        self.base_url = _as_str(self.base_url, "https://ark.cn-beijing.volces.com/api/v3")
        self.model = _as_str(self.model, "doubao-seedream-4-0-250828")
        self.size = _as_str(self.size, "1728x2304")
        self.response_format = _as_str(self.response_format, "url").lower()
        if self.response_format not in {"url", "b64_json"}:
            logger.warning("[生图] response_format={!r} 不认识，改用 url", self.response_format)
            self.response_format = "url"
        self.max_refs = _as_int(self.max_refs, 3, minimum=0)
        self.daily_limit = _as_int(self.daily_limit, 30, minimum=0)
        self.cooldown_seconds = _as_float(self.cooldown_seconds, 60.0)
        self.timeout_seconds = _as_float(self.timeout_seconds, 120.0, minimum=1.0)
        self.output_dir = _as_str(self.output_dir, "generated")
        self.ref_max_edge = _as_int(self.ref_max_edge, 1536, minimum=0)
        self.ref_jpeg_quality = _as_int(self.ref_jpeg_quality, 88, minimum=1)
        self.ref_jpeg_quality = min(95, self.ref_jpeg_quality)
        self.allowed_users = _as_list(self.allowed_users, [], keep_empty=True)
        self.commands = _as_list(self.commands, ["画一张"])
        self.natural_requests = _as_bool(self.natural_requests, default=True)
        self.context_photos = _as_bool(self.context_photos, default=True)

    @property
    def configured(self):
        return bool(self.enabled and self.api_key)

    def allows(self, user):
        """这个用户能不能触发生图。空名单 = 所有人。"""
        if not self.allowed_users:
            return True
        return str(user) in {str(u) for u in self.allowed_users}

    def describe(self):
        """自检那一行用。不打印 Key 本身，只说有没有。"""
        return (f"生图 {'开' if self.enabled else '关'} / "
                f"Key {'已配置' if self.api_key else '未配置'} / "
                f"{self.model} / {self.size} / 每日上限 {self.daily_limit} 张")


def load_config(path=CONFIG_PATH):
    """读 knowledge.toml，返回 (人物设定 dict, 生图设置)。文件缺了就用兜底值。"""
    character, image = dict(FALLBACK_CHARACTER), {}
    try:
        with open(path, "rb") as source:
            raw = tomllib.load(source)
        # 这两个表都不是"必须有"，但如果有就必须是表。
        # 写错成 image = "xxx" 时以前会直接 AttributeError 崩掉整个 load_config。
        section = raw.get("character")
        if isinstance(section, dict):
            character.update({k: v for k, v in section.items() if v})
        elif section is not None:
            logger.warning("[生图] [character] 不是表，已忽略")
        section = raw.get("image")
        if isinstance(section, dict):
            image = section
        elif section is not None:
            logger.warning("[生图] [image] 不是表，已忽略")
    except FileNotFoundError:
        logger.warning("[生图] 找不到 {}，用兜底配置", path)
    except Exception as exc:
        logger.error("[生图] 读取 {} 失败：{}；用兜底配置", path, type(exc).__name__)

    known = {f.name for f in fields(ImageSettings)}
    unknown = set(image) - known
    if unknown:
        # 静默忽略拼错的字段会让人对着配置排查半天
        logger.warning("[生图] [image] 里有不认识的字段，已忽略：{}", "、".join(sorted(unknown)))
    settings = ImageSettings(**{k: v for k, v in image.items() if k in known})
    # 密钥只从环境变量来，不写进配置文件
    settings.api_key = os.environ.get("IMAGE_API_KEY", "").strip()
    settings.base_url = os.environ.get("IMAGE_BASE_URL", "").strip() or settings.base_url
    settings.model = os.environ.get("IMAGE_MODEL", "").strip() or settings.model
    return character, settings


def clean_scene(scene):
    """清洗群友给的场景描述。

    这段文本会原样进提示词，所以必须限制长度、剥掉 CQ 码、挡掉明显不该画的东西。
    这是减速带不是保证：真正的兜底是人物描述永远排在最前、且长度受限。
    返回 (清洗后的场景, 拒绝原因或 None)。
    """
    text = CQ_CODE.sub("", str(scene or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "", None
    if SCENE_BLOCKLIST.search(text):
        return "", "场景描述里有不合适的内容"
    if len(text) > MAX_SCENE_CHARS:
        text = text[:MAX_SCENE_CHARS].rstrip()
    return text, None


def compose_prompt(character, scene=""):
    """拼提示词：固定人物描述 + 场景。场景留空就用配置里的极简室内。

    各段之间补上逗号 —— 直接拼接会变成「看着镜头米白色宽松粗针织毛衣」，
    模型读起来是连在一起的，出图会跑偏。

    人物描述永远排在场景前面：场景是群友输入的、不可信，放前面会把人物带跑。
    """
    parts = [str(character.get(k, "")).strip().rstrip("。，,.")
             for k in ("appearance", "outfit", "style")]
    tail = (scene or "").strip().rstrip("。，,.") or str(character.get("scene", "")).strip()
    if tail:
        parts.append(tail.rstrip("。，,."))
    return "，".join(p for p in parts if p) + "。"


def encode_image(path, *, max_edge=1536, quality=88):
    """参考图转 data URI。火山方舟的 image 字段接受这种写法。

    会先缩到长边 max_edge 并转 JPEG：参考图只用来"认脸"，
    不需要原始 4K 大图。实测 12 张 4MB 的 PNG 直接发，
    3 张就要 16.6MB 请求体，又慢又容易撞服务端上限；
    缩完通常 200~400KB，人物相似度没有可见差别。

    Pillow 不可用时退回原图，不因为少个可选依赖就发不出图。
    """
    raw, mime = _read_for_upload(path, max_edge=max_edge, quality=quality)
    payload = base64.b64encode(raw).decode("ascii")
    return f"data:image/{mime};base64,{payload}"


def _read_for_upload(path, *, max_edge, quality):
    """返回 (图片字节, mime 子类型)。缩不了就原样返回。"""
    suffix = path.suffix.lower().lstrip(".")
    suffix = "jpeg" if suffix == "jpg" else suffix
    if not max_edge:
        return path.read_bytes(), suffix
    try:
        from PIL import Image
    except Exception:
        return path.read_bytes(), suffix
    try:
        from io import BytesIO

        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            longest = max(width, height)
            if longest > max_edge:
                scale = max_edge / longest
                image = image.resize((max(1, round(width * scale)),
                                      max(1, round(height * scale))),
                                     Image.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
    except Exception as exc:
        logger.warning("[生图] 参考图 {} 压缩失败（{}），改用原图", path.name, type(exc).__name__)
        return path.read_bytes(), suffix
    # 压完反而更大就用原图（已经是小图时可能发生）
    if len(data) >= path.stat().st_size:
        return path.read_bytes(), suffix
    return data, "jpeg"


def _natural_key(path):
    """按人类习惯排序文件名：让 (2) 排在 (10) 前面。

    纯字符串排序会把 "10" 排在 "2" 前面（因为 "1" < "2"），
    于是"取前 N 张"实际拿到的是 (1)(10)(11)——完全不是用户想的那个顺序。
    """
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def collect_references(ref_dir, max_refs=3):
    """从参考图目录挑图。

    按文件名自然排序取前 N 张，所以 01_正面.jpg 这种命名能排优先级，
    而 "(2)" 也会正确地排在 "(10)" 前面。
    """
    directory = Path(ref_dir)
    if not directory.is_dir():
        return []
    files = sorted((p for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                   key=_natural_key)
    return files[:max_refs] if max_refs else []


class ImageClient:
    """一次生图 = 一个 HTTP 请求 + 一次下载。失败抛 ImageError，由调用方决定怎么处理。"""

    def __init__(self, character=None, settings=None, *, session=None, clock=time.monotonic):
        if character is None or settings is None:
            loaded_character, loaded_settings = load_config()
            character = character or loaded_character
            settings = settings or loaded_settings
        self.character = character
        self.settings = settings
        self._session = session
        self._lock = asyncio.Lock()
        self._used_today = 0
        self._used_date = date.today()
        # None 而不是 0：clock 从 0 开始时，0 会被当成"没冷却过"
        self._last_at = None
        self._clock = clock

    @classmethod
    def from_config(cls, path=CONFIG_PATH, **kwargs):
        character, settings = load_config(path)
        return cls(character, settings, **kwargs)

    @property
    def configured(self):
        return self.settings.configured

    # ---- 限流：冷却 + 每日张数。群聊的限流管不到生图，必须自己来 ----

    def _roll_date(self):
        today = date.today()
        if today != self._used_date:
            self._used_date = today
            self._used_today = 0

    def quota_left(self):
        self._roll_date()
        return max(0, self.settings.daily_limit - self._used_today)

    def blocked_reason(self, now=None, *, user=None):
        """返回不能生图的原因，能生则返回 None。调用方拿它打日志或回一句话。

        注意：这里只是"看一眼"。真正的额度扣减在 _reserve() 里加锁做，
        否则两个并发请求会都通过检查、把额度打穿。
        """
        if not self.settings.enabled:
            return "生图功能未开启"
        if not self.settings.api_key:
            return "未配置 IMAGE_API_KEY"
        if user is not None and not self.settings.allows(user):
            return "你不在生图白名单里"
        self._roll_date()
        if self._used_today >= self.settings.daily_limit:
            return f"今天的 {self.settings.daily_limit} 张额度用完了"
        now = self._clock() if now is None else now
        if self._last_at is not None:
            wait = self.settings.cooldown_seconds - (now - self._last_at)
            if wait > 0:
                return f"冷却中，还要 {wait:.0f} 秒"
        return None

    def _reserve(self, now):
        """在锁内检查并占额度，避免并发把每日上限打穿。返回拒绝原因或 None。"""
        self._roll_date()
        if self._used_today >= self.settings.daily_limit:
            return f"今天的 {self.settings.daily_limit} 张额度用完了"
        if self._last_at is not None:
            wait = self.settings.cooldown_seconds - (now - self._last_at)
            if wait > 0:
                return f"冷却中，还要 {wait:.0f} 秒"
        self._used_today += 1
        self._last_at = now
        return None

    def _release(self):
        """生图失败时把刚占的额度退回去——失败不该算在用户头上。"""
        self._roll_date()
        self._used_today = max(0, self._used_today - 1)

    # ---- 主流程 ----

    async def generate(self, scene="", *, now=None, user=None):
        """生成一张图，返回落地后的本地路径。任何失败抛 ImageError。"""
        blocked = self.blocked_reason(now, user=user)
        if blocked:
            # 这些原因是"为什么现在不能画"，说给群里听没有副作用
            raise ImageError(blocked, user_safe=True)

        scene, rejected = clean_scene(scene)
        if rejected:
            raise ImageError(rejected, user_safe=True)

        refs = collect_references(self.character.get("ref_dir", "character_refs"),
                                  self.settings.max_refs)
        if not refs:
            logger.warning("[生图] 参考图目录里没有图，退化成纯文生图（人物会不像）")

        prompt = compose_prompt(self.character, scene)
        logger.info("[生图] 准备参考图 {} 张", len(refs))
        # 压缩参考图是同步工作，放到线程里，避免大图阻塞收消息和超时计时。
        payload = await asyncio.to_thread(self._build_payload, prompt, refs)
        logger.info("[生图] 参考图准备完成，等待生图锁")

        async with self._lock:  # 同一时刻只允许一次生图，避免把额度打光
            stamp = now if now is not None else self._clock()
            refused = self._reserve(stamp)
            if refused:
                raise ImageError(refused, user_safe=True)
            try:
                data = await self._post(payload)
                path = await self._save(self._extract_image(data))
            except (Exception, asyncio.CancelledError):
                self._release()
                raise

        logger.info("[生图] 成功，参考图 {} 张，剩余额度 {} 张，已存到 {}",
                    len(refs), self.quota_left(), path.name)
        return path

    def _build_payload(self, prompt, refs):
        """组装请求体。

        image 传数组就是多图参考，正好用来锁人物：给得越多越像，
        但超过 3 张收益递减，还更慢更贵。
        """
        payload = {
            "model": self.settings.model,
            "prompt": prompt,
            "size": self.settings.size,
            # 不传组图参数：部分生图模型连 "disabled" 也不支持，会返回 400。
            "response_format": self.settings.response_format,
            "watermark": False,
        }
        if refs:
            encoded = [encode_image(p, max_edge=self.settings.ref_max_edge,
                                    quality=self.settings.ref_jpeg_quality)
                       for p in refs]
            payload["image"] = encoded[0] if len(encoded) == 1 else encoded
        return payload

    async def _post(self, payload):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
        }
        url = self.settings.base_url.rstrip("/") + "/images/generations"
        session, owned = self._session, self._session is None
        if owned:
            session = aiohttp.ClientSession()
        try:
            timeout = aiohttp.ClientTimeout(total=self.settings.timeout_seconds)
            started = time.monotonic()
            logger.info("[生图] 开始请求接口，模型 {}，超时 {} 秒",
                        self.settings.model, self.settings.timeout_seconds)
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                request_id = (response.headers.get("X-Request-Id")
                              or response.headers.get("X-Tt-Logid") or "未提供")
                logger.info("[生图] 接口响应 HTTP {}，耗时 {:.1f} 秒，Request ID {}；读取响应体",
                            response.status, time.monotonic() - started, request_id)
                body = await response.text()
                if response.status != 200:
                    detail = body.replace(self.settings.api_key, "[已隐藏]")[:2000]
                    raise ImageError(f"生图接口返回 {response.status}，Request ID {request_id}：{detail}")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise ImageError(f"生图接口返回的不是 JSON，Request ID {request_id}") from exc
                logger.info("[生图] 接口响应体读取完成，开始提取图片")
                return data
        except asyncio.TimeoutError as exc:
            raise ImageError(f"生图超时（{self.settings.timeout_seconds:.0f} 秒）") from exc
        except aiohttp.ClientError as exc:
            raise ImageError(f"生图请求失败：{type(exc).__name__}") from exc
        finally:
            if owned:
                await session.close()

    @staticmethod
    def _extract_image(data):
        """取返回里的图片，URL 和 base64 两种都兜住。"""
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items:
            message = None
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                elif error:
                    message = str(error)
            raise ImageError(f"生图返回里没有图片：{message or str(data)[:200]}")
        first = items[0]
        if not isinstance(first, dict):
            raise ImageError("生图返回的图片项格式不对")
        if first.get("url"):
            return {"kind": "url", "value": first["url"]}
        if first.get("b64_json"):
            return {"kind": "b64", "value": first["b64_json"]}
        raise ImageError("生图返回里既没有 url 也没有 b64_json")

    def _output_path(self):
        """给输出文件起个不会撞的名字。

        原来只用秒级时间戳，同一秒内连出两张会互相覆盖（第二张把第一张冲掉）。
        这里带上微秒，并且真去查一下有没有撞上。
        """
        directory = Path(self.settings.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for attempt in range(100):
            stamp = f"{datetime.now():%Y%m%d-%H%M%S-%f}"[:-3]
            path = directory / f"{stamp}{'' if not attempt else f'-{attempt}'}.png"
            if not path.exists():
                return path
        raise ImageError("生成的文件名一直撞车，请检查 output_dir")

    async def _save(self, raw):
        """统一落地成本地文件。

        必须自己下载：接口给的 URL 只有 24 小时有效，
        而 NapCat 发本地文件比让它去外网拉图稳得多。
        """
        path = self._output_path()

        if raw["kind"] == "b64":
            logger.info("[生图] 开始保存 base64 图片")
            try:
                path.write_bytes(base64.b64decode(raw["value"]))
            except (binascii.Error, ValueError) as exc:
                # 以前这里会漏出 binascii.Error，调用方只 catch ImageError，直接炸穿
                raise ImageError("生图返回的 base64 图片解不开") from exc
            return path

        session, owned = self._session, self._session is None
        if owned:
            session = aiohttp.ClientSession()
        try:
            timeout = aiohttp.ClientTimeout(total=self.settings.timeout_seconds)
            logger.info("[生图] 开始下载生成图片，超时 {} 秒", self.settings.timeout_seconds)
            async with session.get(raw["value"], timeout=timeout) as response:
                logger.info("[生图] 图片下载响应 HTTP {}，读取图片内容", response.status)
                if response.status != 200:
                    raise ImageError(f"下载生成的图片失败：{response.status}")
                path.write_bytes(await response.read())
        except asyncio.TimeoutError as exc:
            raise ImageError(f"下载生成的图片超时（{self.settings.timeout_seconds:.0f} 秒）") from exc
        except aiohttp.ClientError as exc:
            raise ImageError(f"下载生成的图片失败：{type(exc).__name__}") from exc
        finally:
            if owned:
                await session.close()
        return path

    async def close(self):
        """只关自己开的会话；注入进来的会话由注入方负责。"""
        if self._session is not None and not self._session.closed:
            await self._session.close()


def build_image_segment(path):
    """把本地图片包成 OneBot 消息段。

    只有本函数构造的图片段才会被发送 —— 模型生成的文本里就算写了 CQ 码，
    也会在 ai_service._clean() 里被剥掉。这条边界不要为了省事放宽。
    """
    return {"type": "image", "data": {"file": Path(path).resolve().as_uri()}}


def _selftest():
    """python image_service.py —— 打印配置状态，不真的生图。"""
    character, settings = load_config()
    print("=" * 60)
    print("生图模块自检")
    print("=" * 60)
    print(settings.describe())
    print(f"接口地址：{settings.base_url}/images/generations")
    refs = collect_references(character["ref_dir"], settings.max_refs)
    print(f"参考图：{character['ref_dir']} → 共 {len(refs)} 张参与上传")
    raw_total = sum(item.stat().st_size for item in refs)
    sent_total = 0
    for item in refs:
        uri = encode_image(item, max_edge=settings.ref_max_edge,
                           quality=settings.ref_jpeg_quality)
        sent_total += len(uri)
        print(f"  - {item.name}")
        print(f"      {item.stat().st_size / 1024:6.0f} KB  →  上传 {len(uri) / 1024:5.0f} KB")
    if refs:
        print(f"  请求体合计约 {sent_total / 1024 / 1024:.2f} MB"
              f"（原始 {raw_total / 1024 / 1024:.2f} MB，"
              f"压缩到 {sent_total / max(raw_total, 1) * 100:.0f}%）")
    if not refs:
        print("  ⚠ 目录为空。放 2~3 张角色照片进去（命名 01_正面.jpg / 02_侧面.jpg / 03_全身.jpg），")
        print("    否则退化成纯文生图，人物会不像。")
    else:
        print("  提示：参考照片自己带的文字/水印会被模型当成画面的一部分学过去，")
        print("        尽量用没有水印、没有文字的照片。")
    print("-" * 60)
    print("实际会发出去的提示词：")
    print(compose_prompt(character, ""))
    print("-" * 60)
    print(f"触发词：{'、'.join(settings.commands)}")
    print(f"可用人群：{'所有人' if not settings.allowed_users else '、'.join(settings.allowed_users)}")
    print("-" * 60)
    if settings.configured:
        print("配置完整，可以生图。")
    else:
        missing = []
        if not settings.enabled:
            missing.append("knowledge.toml 里 [image] enabled = true")
        if not settings.api_key:
            missing.append(".env 里 IMAGE_API_KEY=...")
        print("还差：" + "；".join(missing))


if __name__ == "__main__":
    _selftest()
