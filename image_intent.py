"""识别明确的中文出图请求；只解析当前消息，不让聊天回复自行触发付费调用。"""

import re


_POLITE = re.compile(
    r"^(?:请你|请|麻烦你|麻烦|拜托你|拜托|帮我|给我|让我|替我|"
    r"能不能|可不可以|能否|可以|能|方便的话|现在|再|先|你)"
)
_PHOTO = (r"(?:自拍照?|照片|相片|写真|图片|"
          r"图(?=$|[，,：:。.!！?？\s]|给我|看看|看一下|吧|呗|呀|啊|呢|嘛|吗|啦))")
_REQUEST = re.compile(
    rf"^(?:来|发给我|发我|发|拍给我|拍)(?:一?[张个])?"
    rf"(?P<before>.{{0,60}}?)(?P<photo>{_PHOTO})(?P<after>.*)$"
)
_WANT = re.compile(
    rf"^(?:我)?(?:想看看|想看|想要|要看|看看|看一下)"
    rf"(?P<before>.{{0,60}}?)(?P<photo>{_PHOTO})(?P<after>.*)$"
)
_DRAW = re.compile(r"^(?:画|生成|制作|做|出)(?:一?[张幅个])(?P<scene>.*)$")
# 只匹配看角色本人的请求，不把“看看你写的代码”等讨论当成照片请求。
_SEE_YOU = re.compile(
    r"^(?:我)?(?:想看看|想看|想见见|想见|看看|看一下|见见)"
    r"(?:你本人|你现在的样子|你现在什么样|你现在的样貌|你|本人)"
    r"(?:一下|的样子|长什么样)?(?:好不好|可以吗|行吗|好吗|吗|嘛|吧|呗|呀|啊|呢)*$"
)
_NEGATIVE = re.compile(
    rf"(?:(?<!特)别|不要|不用|不必|不想|不需要|禁止).{{0,16}}(?:{_PHOTO}|画|生成|生图|拍|发)"
    r"|(?:取消|停止)(?:生图|生成|画图|出图|发图|自拍)"
)
_DISCUSSION = re.compile(
    r"怎么|如何|教程|触发词|触发条件|什么意思|啥意思|是什么|是什么东西|"
    r"能做什么|支持什么|会不会|会画|会生成|能不能画图|能画图|"
    r"这句话|这几个字|这条指令|这个命令|功能|是什么意思|多少钱|收费|计费"
)
_QUOTED = re.compile(r"[“”「」『』\"`]|^(?:他说|她说|别人说|有人说|例如|比如|如果|假如)")
_ENDING = re.compile(
    r"(?:给我看看|给我看|看看|看一下|瞧瞧|好不好|可以吗|行不行|行吗|好吗|"
    r"可不可以|给我|一下|吧|呗|呀|啊|呢|嘛|吗|啦)$"
)


def _scene(text):
    text = text.strip(" ，,：:。.!！?？~～")
    while True:
        cleaned = _ENDING.sub("", text).rstrip(" ，,。.!！?？~～")
        if cleaned == text:
            break
        text = cleaned
    # 人物身份由参考图决定；这些代词不属于场景。
    text = re.sub(r"^(?:你自己|自己的|自己|你的|你|一张|的)+", "", text)
    return text.strip(" 的，,：:")


def match_image_request(text, commands, *, natural=True, names=()):
    """返回 (场景, 是否请求)。否定、引用、用法/能力讨论优先排除。"""
    candidate = str(text).strip()
    if not candidate:
        return "", False
    if natural:
        if (_NEGATIVE.search(candidate) or _DISCUSSION.search(candidate)
                or _QUOTED.search(candidate)):
            return "", False
        for name in sorted((name for name in names if name), key=len, reverse=True):
            if candidate.startswith(name):
                candidate = candidate[len(name):].lstrip(" ，,：:！!~～")
                break
        while True:
            cleaned = _POLITE.sub("", candidate, count=1).lstrip(" ，,：:")
            if cleaned == candidate:
                break
            candidate = cleaned

    if natural:
        if _SEE_YOU.fullmatch(candidate.strip(" ，,。.!！?？~～")):
            return "", True
        draw = _DRAW.match(candidate)
        if draw:
            scene = re.sub(r"^(?:图片|图像|自拍照?|照片|相片|图)(?:[，,：:]|$)", "", draw["scene"])
            scene = _scene(scene)
            scene = re.sub(r"(?:的)?(?:图片|图像|自拍照?|照片|相片)$", "", scene)
            return _scene(scene), True
        for pattern in (_REQUEST, _WANT):
            request = pattern.match(candidate)
            if request:
                before, after = _scene(request["before"]), _scene(request["after"])
                return "，".join(part for part in (before, after) if part), True

    # 自定义触发词仍可用，长词优先，避免短词把长词的尾巴误当场景。
    for command in sorted(commands, key=len, reverse=True):
        if candidate.startswith(command):
            return _scene(candidate[len(command):]), True
    return "", False
