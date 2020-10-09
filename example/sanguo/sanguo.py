from reb import P

# 通用模式

# 标点符号
punc = P.ic('，。：；“”？！【】「」、‘’')

# 非标点符号
nopunc = P.nic('，。：；“”？！【】「」、‘’\n\r\t \u3000')

# 发语词
fyc = P.any('却说', '且说', '话说', '盖', '原来')

# 逗号
comma = P.pattern('，')

# 无实义
wsy = P.any('之', '乎', '者', '也')


# 姓名相关模式

# 姓名
name = P.tag(
    # 中文姓名在 2 到 4 字之间
    P.n(nopunc, 2, 4, greedy=False),
    tag='姓名')


# 字
zi = (
    '字' + P.tag(
        # 中文字一般两个汉字
        P.n(nopunc, exact=2),
        tag='字'
    )
)

# 姓
xing = (
    (  # 覆姓某某
        '覆姓' + P.tag(P.n(nopunc, exact=2), tag='姓')
    ) | (  # 姓某
        '姓' + P.tag(nopunc, tag='姓')
    )
)


# 描述姓名的语句片段
name_clause = (
    '姓甚名谁'  # 姓甚名谁是特殊用法，不能提出真实姓名
    | P.example(
        xing + P.n01(comma) + '名' + P.tag(P.n(nopunc, 1, 2, greedy=False), tag='名') + P.n01(P.n01(comma) + zi),
            '姓刘名备字玄德',
            '姓郭名攸之',  # 两个汉字当作名字
            '姓刘，名备，字玄德',  # 用逗号分割的
            '姓刘，名备',  # 没有说字
            '覆姓太史，名慈，字子义',  # 覆姓
    ) | P.example(
        name + P.n01(comma) + zi,
            '刘备，字玄德',
            '司马懿，字仲达',
    )
)


# 地点，一般是两个汉字
place = P.tag(
    P.n(nopunc, exact=2),
    tag='地点'
)


# 人物籍贯子句
place_clause = (
    P.n01('乃') + place + P.n01(place) + '人' + P.n01('也')
)


# 主模式
pattern = (
    P.n01(fyc)  # 可以接发语词
    + (
        P.onceeach(name_clause, place_clause, seperator=punc)
        | name_clause
    )
    + P.n(wsy)  # 可以接无实义字
    + punc  # 子句以标点结束
)
