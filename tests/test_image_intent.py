import unittest

from image_intent import match_image_request


COMMANDS = ["画一张", "来个自拍", "生成一张", "发张照片", "发个照片", "看看照片"]


class ImageIntentTests(unittest.TestCase):
    def test_natural_requests_and_scenes(self):
        cases = {
            "来张自拍": "",
            "来个自拍": "",
            "帮我画一张在咖啡馆看书": "在咖啡馆看书",
            "能不能给我发一张你的照片呀？": "",
            "你能发张你的照片吗": "",
            "请发一张你在咖啡馆的自拍给我看看好吗": "在咖啡馆",
            "我想看看你在海边的照片": "在海边",
            "想看你的自拍": "",
            "给我看看你的照片吧": "",
            "拍张自拍，穿白色毛衣": "穿白色毛衣",
            "麻烦你帮我画张在图书馆的照片": "在图书馆",
            "甜甜，来张自拍吧": "",
            "发张照片看看": "",
            "看看照片": "",
            "生成一张图片：在咖啡馆": "在咖啡馆",
            "发张自拍，不要水印": "不要水印",
            "发张你在图书馆的自拍": "在图书馆",
            "想看看你在图书馆的照片": "在图书馆",
            "来张图": "",
            "来张特别好看的自拍": "特别好看",
            "看看你": "",
            "我想看看你": "",
            "让我看看你吧": "",
            "能让我看看你现在的样子吗": "",
        }
        for text, scene in cases.items():
            with self.subTest(text=text):
                self.assertEqual(match_image_request(text, COMMANDS, names=("甜甜",)), (scene, True))

    def test_discussion_negation_and_quotes_do_not_spend_quota(self):
        cases = [
            "你会画图吗", "你能生成图片吗", "自拍是什么", "你有照片吗",
            "别发照片", "不要给我发自拍", "不用画一张", "我不想看你的自拍",
            "来张自拍，但是不要生成", "画一张的触发条件是什么", "发张照片怎么收费",
            "发张照片这句话是什么意思", "他说来张自拍", "如果我说来张自拍会怎样",
            "“来张自拍”", "帮我解释一下来个自拍", "今天发照片了吗",
            "这张照片真好看", "来个自拍是什么意思", "发张照片的教程", "来张自拍多少钱",
            "你在哪", "你在干嘛", "看看你写的代码", "看看你的计划", "看看你说的链接",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(match_image_request(text, COMMANDS), ("", False))

    def test_custom_commands_and_opt_out(self):
        self.assertEqual(match_image_request("晒一张在花园", ["晒一张"]), ("在花园", True))
        self.assertEqual(match_image_request("来张自拍", COMMANDS, natural=False), ("", False))
        self.assertEqual(match_image_request("帮我画一张", COMMANDS, natural=False), ("", False))
        self.assertEqual(match_image_request("画一张在海边", COMMANDS, natural=False), ("在海边", True))
        self.assertEqual(match_image_request("来个自拍照", ["来个自拍", "来个自拍照"]), ("", True))
