#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared story data for《嘟嘟的一天》— shot order, dialogue lines and title.

Text lives here; timing is derived (see `gen_subs.py`). Kept free of any
model/ffmpeg imports so every stage can use it.
"""
from __future__ import annotations

TITLE_ZH = "嘟嘟的一天"
TITLE_EN = "Dodo's Day"

ORDER = [
    "01_wakeup", "02_wakeup_talk", "03_breakfast", "04_school_road", "05_classroom",
    "06_naptime", "07_playground", "08_dig", "09_pond", "10_splash", "11_swim",
    "12_dinner", "13_eat", "14_bedtime", "15_sleep",
]

# one entry per spoken line in the shot, in order: (中文, English)
SHOT_LINES = {
    "01_wakeup": [
        ("清晨，猪圈里的小猪嘟嘟还在做着甜甜的梦。",
         "Early in the morning, little pig Dudu is still dreaming sweet dreams."),
    ],
    "02_wakeup_talk": [
        ("妈妈猪：嘟嘟！太阳晒屁股啦！再不起来，早饭就变成晚饭了！",
         "Mama Pig: Dudu! The sun is up! Get up, or breakfast will turn into dinner!"),
        ("嘟嘟：呜……还想再睡五分钟……好啦好啦……我起来了……",
         "Dudu: Mmm... five more minutes... Okay, okay... I'm up..."),
    ],
    "03_breakfast": [
        ("妈妈猪：今天有热乎乎的玉米粥和苹果片！",
         "Mama Pig: Today we have warm corn porridge and apple slices!"),
        ("嘟嘟：香香香！", "Dudu: Yummy, yummy!"),
    ],
    "04_school_road": [
        ("咩咩：嘟嘟早！你鼻子上还沾着粥呢！",
         "Lamb: Morning, Dudu! You still have porridge on your nose!"),
        ("嘟嘟：嘿嘿……好吃的留下的痕迹！走吧走吧！",
         "Dudu: Hehe... a trace of something tasty! Let's go!"),
    ],
    "05_classroom": [
        ("老师：今天我们学习“分享”……嘟嘟，你为什么一直在拱桌脚？",
         "Teacher: Today we learn about sharing... Dudu, why are you pushing the desk leg?"),
        ("嘟嘟：老师，我在思考！思考怎么把好吃的分享给大家！",
         "Dudu: Teacher, I'm thinking! Thinking how to share the tasty food with everyone!"),
    ],
    "06_naptime": [
        ("嘟嘟：汪汪……我好想睡觉……", "Dudu: Woof-woof... I'm so sleepy..."),
        ("汪汪：又开始做泥巴梦了……", "Puppy: He's having his mud dream again..."),
    ],
    "07_playground": [
        ("老师：今天自由活动！", "Teacher: Free play time today!"),
        ("嘟嘟：我要比赛谁拱得最快！", "Dudu: I want to see who can dig the fastest!"),
    ],
    "08_dig": [
        ("咩咩：嘟嘟你又把操场拱成迷宫了！",
         "Lamb: Dudu, you've turned the playground into a maze again!"),
    ],
    "09_pond": [
        ("嘎嘎：来游泳呀！", "Duckling: Come swim!"),
        ("嘟嘟：凉凉的……", "Dudu: It's cool..."),
    ],
    "10_splash": [
        ("嘟嘟：噗噗噗！我是水里的小火箭！",
         "Dudu: Splash, splash! I'm a little rocket in the water!"),
    ],
    "11_swim": [
        ("旁白：大家在池塘里玩得可开心啦。",
         "Narrator: Everyone is having so much fun in the pond."),
    ],
    "12_dinner": [
        ("妈妈猪：今天玩得开心吗？", "Mama Pig: Did you have fun today?"),
        ("嘟嘟：开心！我拱了操场、游了泳、还差点把桌子拱走！",
         "Dudu: Yes! I dug up the playground, swam, and almost pushed the table away!"),
    ],
    "13_eat": [
        ("妈妈猪：那明天继续努力……先把晚饭吃完！",
         "Mama Pig: Then keep it up tomorrow... finish your dinner first!"),
        ("嘟嘟：嗯嗯……今天的南瓜最好吃！", "Dudu: Mmm... today's pumpkin is the tastiest!"),
    ],
    "14_bedtime": [
        ("嘟嘟：妈妈……明天还要上学吗？", "Dudu: Mama... do we have school again tomorrow?"),
        ("妈妈猪：要的。", "Mama Pig: Yes."),
    ],
    "15_sleep": [
        ("嘟嘟：那我……明天还要……拱……", "Dudu: Then I... tomorrow I'll still... dig..."),
        ("旁白：就这样，嘟嘟带着甜甜的梦，睡着了。",
         "Narrator: And so, with sweet dreams, Dudu fell asleep."),
    ],
}
